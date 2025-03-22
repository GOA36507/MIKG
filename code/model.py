import gc
import time
import tensorflow as tf
import numpy as np
from layer import RGATLayer
from util import embed_init
from test_funcs import greedy_alignment
import math
import logging
import random

eps = 1e-7

class NCECriterion(object):
    def __init__(self, n_data):
        self.n_data = float(n_data)

    def forward(self, x):
        # x: Tensor with shape [batch_size, K+1]
        with tf.name_scope("NCECriterion"):
            x = tf.convert_to_tensor(x)
            bsz = tf.shape(x)[0]
            m = tf.shape(x)[1] - 1
            Pn = 1.0 / self.n_data
            P_pos = x[:, 0]
            log_D1 = tf.log(P_pos / (P_pos + tf.cast(m, tf.float32) * Pn + eps))
            P_neg = x[:, 1:]
            log_D0 = tf.log((tf.cast(m, tf.float32) * Pn) / (P_neg + tf.cast(m, tf.float32) * Pn + eps))
            loss = - (tf.reduce_sum(log_D1) + tf.reduce_sum(log_D0)) / tf.cast(bsz, tf.float32)
            return loss

class NCE_Estimator(object):
    def __init__(self, temperature=1.0, weight=1.0):
        self.temperature = temperature
        self.weight = weight
        self.estimator_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    def forward(self, input_embeds, all_embeds, labels):
        """
        input_embeds: [batch_size, dim]
        all_embeds: [entity_num, dim]
        labels: [batch_size] (index of positives in all_ent)
        """
        sim_matrix = tf.matmul(input_embeds, tf.transpose(all_embeds)) / self.temperature
        estimator = self.estimator_fn(logits=sim_matrix, labels=labels)
        return tf.reduce_mean(estimator) * self.weight

class MIKG:
    def __init__(self, kgs, ent_adj, ent_rel_adj, params,
                 value_embedding, ent_embedding, attribute_embedding, relation_embedding):
        self.kgs = kgs
        self.params = params
        self.rel_flag = False
        self.ent_num = kgs.entities_num
        self.value_num = kgs.values_num
        self.attr_num = kgs.attributes_num
        self.rel_num = kgs.relations_num
        self.ents1 = kgs.useful_entities_list1
        self.ents2 = kgs.useful_entities_list2
        self.sup_links = kgs.train_links
        self.ref_links = kgs.test_links
        self.valid_links = kgs.valid_links
        self.attr_value_list = kgs.kg1.attr_value_list + kgs.kg2.attr_value_list
        self.value_list = kgs.value_list
        self.train_entities1 = kgs.train_entities1
        self.train_entities2 = kgs.train_entities2
        self.valid_entities1 = kgs.valid_entities1
        self.valid_entities2 = kgs.valid_entities2
        self.test_entities1 = kgs.test_entities1
        self.test_entities2 = kgs.test_entities2
        self.layer_num = 1
        self.select_attr_K = params.select_attr_K 
        self.value_attr_concate = kgs.value_attr_concate
        if params.model == 'Only Attr':
            self.layer_num = 0
        self.ent_adj = ent_adj
        self.ent_rel_adj = ent_rel_adj
        self.activation = tf.nn.leaky_relu
        self.layers = list()
        self.output = list()
        self.dim = params.dim 
        self.input_value = value_embedding
        if params.model == 'Only Rel':
            self.input_value = None  
        else:
            self.value_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.value_num + 1, 768])
        self.ent_embeddings = ent_embedding
        self.temp_attribute_embeddings = attribute_embedding
        self.temp_relation_embeddings = relation_embedding
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.session = tf.Session(config=config)
        self.lr = self.params.learning_rate
        self.generate_variables()
        if params.model != 'Only Rel':
            self.generate_value_entity_embeddings()
        self.build_training_graph()
        tf.global_variables_initializer().run(session=self.session)
        if self.input_value is not None:
            self.session.run(tf.assign(self.init_value_embeddings, self.value_placeholder),
                             {self.value_placeholder: self.input_value})

    def generate_variables(self):
        self.ent_padding = tf.constant(0, dtype=tf.float32, shape=(1, self.params.dim))
        if self.input_value is not None:
            self.rel_flag = True
            self.no_attr = embed_init(1, 768, "no_see_attr", method='glorot_uniform_initializer')
            with tf.variable_scope("value_embeddings"):
                self.init_value_embeddings = tf.get_variable('empty_value',
                                                            dtype=tf.float32,
                                                            shape=[self.value_num + 1, 768])
            with tf.variable_scope("relation_embeddings"):
                self.rel_embeddings = tf.Variable(self.temp_relation_embeddings, trainable=True, dtype=tf.float32)
                rel_padding = tf.constant(0, dtype=tf.float32, shape=(1, 768))
                self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)
                self.rel_embeddings = tf.concat((self.rel_embeddings, rel_padding), axis=0)
            with tf.variable_scope("temp_attribute_embeddings"):
                self.temp_attribute_embeddings = tf.Variable(self.temp_attribute_embeddings,
                                                            trainable=True, dtype=tf.float32)
            with tf.variable_scope("temp_rel_map"):
                self.temp_rel_map = tf.get_variable('rel_map',
                                                    dtype=tf.float32,
                                                    shape=[768, self.params.dim],
                                                    initializer=tf.initializers.glorot_normal(dtype=tf.float32))
        else:
            # 不使用属性拼接时，直接初始化实体 + 数值嵌入
            with tf.variable_scope("ent_embeddings"):
                self.ent_embeddings = tf.get_variable('ent_embedding',
                                                    dtype=tf.float32,
                                                    shape=[self.ent_num, self.params.dim],
                                                    initializer=tf.initializers.glorot_normal(dtype=tf.float32))
            with tf.variable_scope("value_embeddings"):
                temp_value_embeddings = embed_init(self.value_num, self.params.dim, "value_none_zero",
                                                method='glorot_uniform_initializer')
                zero_embeddings = tf.constant(0, dtype=tf.float32, shape=(1, self.params.dim))
                temp_value_embeddings = tf.nn.l2_normalize(temp_value_embeddings, axis=1)
                self.init_value_embeddings = tf.concat((temp_value_embeddings, zero_embeddings), axis=0)
            with tf.variable_scope("relation_embeddings"):
                self.rel_embeddings = tf.get_variable('rel_embedding',
                                                    dtype=tf.float32,
                                                    shape=[self.rel_num, self.params.dim],
                                                    initializer=tf.initializers.glorot_normal(dtype=tf.float32))
                self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)
                zero_embeddings = tf.constant(0, dtype=tf.float32, shape=(1, self.params.dim))
                self.rel_embeddings = tf.concat((self.rel_embeddings, zero_embeddings), axis=0)


    def generate_value_entity_embeddings(self):
        """合并属性和属性值嵌入加权融合为一个函数"""

        attr_embeddings = self.temp_attribute_embeddings
        concate_embeddings = tf.nn.embedding_lookup(attr_embeddings, self.value_attr_concate)
        concate_embeddings = tf.concat((concate_embeddings, self.no_attr), axis=0)
        concat = tf.concat((concate_embeddings, self.init_value_embeddings), axis=1)
        dense_layer = tf.keras.layers.Dense(self.params.dim, use_bias=False)
        self.value_embeddings = tf.nn.l2_normalize(dense_layer(concat), axis=1)
        self.first_value_list = [val[0] for val in self.value_list]
        self.other_value_list = [val[1:self.select_attr_K] for val in self.value_list]
        self.first_value_embeding = tf.nn.embedding_lookup(self.value_embeddings, self.first_value_list)
        value_embeddings = tf.nn.embedding_lookup(self.value_embeddings, self.other_value_list)
        value_mask = tf.cast(tf.not_equal(self.other_value_list, self.value_num), tf.float32)
        value_mask_exp = tf.expand_dims(value_mask, axis=-1)
        value_embeddings = value_embeddings * value_mask_exp  
        ent_tile = tf.tile(tf.expand_dims(self.first_value_embeding, 1), [1, self.select_attr_K-1, 1])
        fused = value_embeddings + ent_tile  
        fused_flat = tf.reshape(fused, [-1, self.params.dim])
        hidden = tf.keras.layers.Dense(128, activation='relu')(fused_flat)
        logits = tf.keras.layers.Dense(1)(hidden)
        logits = tf.reshape(logits, [self.ent_num, self.select_attr_K-1])
        logits_masked = tf.where(value_mask > 0, logits, tf.fill(tf.shape(logits), -1e15))
        attention = tf.nn.softmax(logits_masked, axis=1)  
        attention = tf.expand_dims(attention, axis=-1)    
        weighted_sum_embeddings= tf.reduce_sum(attention * value_embeddings, axis=1) 
        self.ent_embedings = weighted_sum_embeddings + self.first_value_embeding
        with tf.control_dependencies([
            tf.assert_equal(tf.reduce_any(tf.is_inf(self.ent_embeddings)), False, ["ent_embeddings has inf"]),
            tf.assert_equal(tf.reduce_any(tf.is_nan(self.ent_embeddings)), False, ["ent_embeddings has NaN"])
        ]):
            self.ent_embeddings = tf.identity(self.ent_embeddings)
        self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)

    
    def rgat_graph_convolution(self, evaluation=False):
        self.output = list()  # reset
        self.rel_output = list()
        self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)
        output_embeddings = self.ent_embeddings
        rel_embeddings = self.rel_embeddings
        self.output.append(output_embeddings)
        self.rel_output.append(rel_embeddings)
        if not evaluation:
            output_embeddings = tf.nn.dropout(output_embeddings, rate=self.params.input_drop_rate)
        for i in range(self.layer_num):
            if not evaluation:
                activation = None
                if i == self.layer_num - 1:
                    activation = None
                rgat_layer = RGATLayer(self.params.dim, self.params.dim, self.params.dim, self.params.dim,
                                       self.ent_rel_adj, self.ent_adj, self.params.drop_rate, i, self.ent_num,
                                       self.params.neighbor_num, activation)
                self.layers.append(rgat_layer)
            else:
                rgat_layer = self.layers[i]
            output_embeddings = rgat_layer.call(
                output_embeddings, rel_embeddings)
            output_embeddings = tf.nn.l2_normalize(output_embeddings, axis=1)
            rel_embeddings = tf.nn.l2_normalize(rel_embeddings, axis=1)
            self.output.append(output_embeddings)
            self.rel_output.append(rel_embeddings)


    def build_training_graph(self):
        # placeholders
        self.loss = 0.0
        self.input_entities1 = tf.placeholder(tf.int32, shape=[None])
        self.input_entities2 = tf.placeholder(tf.int32, shape=[None])
        self.ce_label = tf.placeholder(tf.float32, shape=[None, None])
        self.mi_label = tf.placeholder(tf.int32, shape=[None])
        if self.params.model != 'Only Rel':
            self.generate_value_entity_embeddings()
        self.ent_embeddings = tf.concat((self.ent_embeddings, self.ent_padding), axis=0)
        if self.rel_flag:
            self.rel_embeddings = tf.matmul(self.rel_embeddings, self.temp_rel_map)
            self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)
        self.attr_embed = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        self.rgat_graph_convolution()
        self.rel_embed = self.output[-1]
        # MI Estimator
        self.nce_estimator = NCE_Estimator(temperature=self.params.mi_t,weight=self.params.mi_w)
        input_embeds_att = tf.nn.embedding_lookup(self.attr_embed, self.input_entities1)
        input_embeds_rel = tf.nn.embedding_lookup(self.rel_embed, self.input_entities1)
        self.loss +=  self.nce_estimator.forward(input_embeds_att, self.attr_embed, self.mi_label) 
        self.loss +=  self.nce_estimator.forward(input_embeds_rel, self.rel_embed, self.mi_label) 
        # Basic CE Loss
        all_ent_embeddings = tf.concat([self.attr_embed, self.rel_embed], axis=1)
        all_ent_embeddings = tf.nn.l2_normalize(all_ent_embeddings, axis=1)
        input_embeds = tf.nn.embedding_lookup(all_ent_embeddings, self.input_entities1)
        self.loss += self.ce_loss(input_embeds, self.ce_label, all_ent_embeddings)
        # Optimizer
        opt = tf.train.AdamOptimizer(self.lr)
        self.optimizer = opt.minimize(self.loss)


    def ce_loss(input_embeds, label, embeds):
        sim = tf.matmul(input_embeds, tf.transpose(embeds))
        sim = tf.sigmoid(sim)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=sim)
        loss = tf.reduce_mean(loss)
        return loss
    
    def train(self, iteration, num_epochs, params, kgs):

        total_time = 0.0
        for epoch in range(num_epochs):
            steps = math.ceil(len(kgs.train_links) / params.batch_size)
            batch_size = math.ceil(len(self.train_entities1) / steps)
            for step in range(steps):
                links = random.sample(self.sup_links, batch_size)
                mi_label = [link[1] for link in links]
                ce_label = (params.neg_pro / kgs.entities_num) * np.ones((batch_size, kgs.entities_num + 1))
                for i, link in enumerate(links):
                    ce_label[i][link[1]] += (1 - params.neg_pro)
                    ce_label[i][-1] = 0
                entities1 = [p[0] for p in links]
                entities2 = [p[1] for p in links]
                feed_dict = {
                    self.input_entities1: entities1,
                    self.input_entities2: entities2,
                    self.ce_label: ce_label,
                    self.mi_label: mi_label
                }
                start = time.time()
                fetches = {
                    "loss": self.loss,
                    "train_op": self.optimizer,
                }
                results = self.session.run(fetches=fetches, feed_dict=feed_dict)
                loss = results["loss"]
                duration = round(time.time() - start, 2)
                total_time += duration
                logging.info("Epoch {:d} Step {:d}: loss = {:.6f}, time = {:.2f}s".format(epoch+1, step+1, loss, duration))
        avg_time = round(total_time / num_epochs, 5) if num_epochs > 0 else 0
        logging.info("average time per epoch = {:.3f} s".format(avg_time))

    def test(self):
        ti = time.time()
        self.rgat_graph_convolution(evaluation=True)
        embeds_list1, embeds_list2 = list(), list()
        for output_embeds in self.output:
            embeds1 = tf.nn.embedding_lookup(output_embeds, self.test_entities1)
            embeds2 = tf.nn.embedding_lookup(output_embeds, self.test_entities2)
            embeds1 = tf.nn.l2_normalize(embeds1, 1)
            embeds2 = tf.nn.l2_normalize(embeds2, 1)
            embeds_list1.append(embeds1)
            embeds_list2.append(embeds2)
        test_embeds1 = tf.concat(embeds_list1, axis=1)
        print("Shape of test_embeds1:", test_embeds1.shape)
        test_embeds2 = tf.concat(embeds_list2, axis=1)
        test_embeds1 = tf.nn.l2_normalize(test_embeds1, axis=1)
        test_embeds2 = tf.nn.l2_normalize(test_embeds2, axis=1)
        test_embeds1 = test_embeds1.eval(session=self.session)
        test_embeds2 = test_embeds2.eval(session=self.session)
        alignment_rest, hits1, mr_12, mrr_12, hits5= greedy_alignment(test_embeds1,
                                                                test_embeds2,
                                                                self.params.ent_top_k,
                                                                self.params.nums_threads,
                                                                'inner', False, 0, True)
        print("test totally costs {:.1f} s ".format(time.time() - ti))
        del test_embeds1, test_embeds2
        gc.collect()
        return  mrr_12, hits1, hits5

    def valid(self):
        ti = time.time()
        embeds_list1, embeds_list2 = list(), list()
        self.rgat_graph_convolution(evaluation=True)
        for output_embeds in self.output:
            embeds1 = tf.nn.embedding_lookup(output_embeds, self.valid_entities1)
            embeds2 = tf.nn.embedding_lookup(output_embeds, self.valid_entities2 + self.test_entities2)
            embeds1 = tf.nn.l2_normalize(embeds1, 1)
            embeds2 = tf.nn.l2_normalize(embeds2, 1)
            embeds_list1.append(embeds1)
            embeds_list2.append(embeds2)
        valid_embeds1 = tf.concat(embeds_list1, axis=1)
        valid_embeds2 = tf.concat(embeds_list2, axis=1)
        valid_embeds1 = tf.nn.l2_normalize(valid_embeds1, axis=1)
        valid_embeds2 = tf.nn.l2_normalize(valid_embeds2, axis=1)
        valid_embeds1 = valid_embeds1.eval(session=self.session)
        valid_embeds2 = valid_embeds2.eval(session=self.session)
        alignment_rest, hits1, mr_12, mrr_12, hits5 = greedy_alignment(valid_embeds1,
                                                                valid_embeds2,
                                                                self.params.ent_top_k,
                                                                self.params.nums_threads,
                                                                'inner', False, 0, True)
        print("validation costs {:.1f} s ".format(time.time() - ti))
        del valid_embeds1, valid_embeds2
        gc.collect()
        return   mrr_12

    def save(self):
        self.rgat_graph_convolution(evaluation=True)
        output = list()
        for output_embeds in self.output:
            embeds = output_embeds
            embeds = tf.nn.l2_normalize(embeds, axis=1)
            output.append(embeds)
        embeds = tf.concat(output, axis=1)
        embeds = tf.nn.l2_normalize(embeds, axis=1)
        first_embeds = output[0]
        first_embeds = tf.nn.l2_normalize(first_embeds, axis=1)
        first_embeds = first_embeds.eval(session=self.session)
        embeds = embeds.eval(session=self.session)
        dataset_name = self.params.input.split("/")[-2]
        name = "../"+dataset_name+"_"+str(self.layer_num)+".npy"
        np.save(name, embeds)