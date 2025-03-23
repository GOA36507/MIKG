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
    def __init__(self, kgs, entity_adj, relation_adj, args,
                 value_embs, entity_embs, attr_embs, rel_embs):
        self.kgs = kgs
        self.args = args
        self.use_rel = False

        self.entity_num = kgs.entities_num
        self.value_num = kgs.values_num
        self.attr_num = kgs.attributes_num
        self.rel_num = kgs.relations_num

        self.train_links = kgs.train_links
        self.valid_links = kgs.valid_links
        self.test_links = kgs.test_links

        self.train_entities1 = kgs.train_entities1
        self.train_entities2 = kgs.train_entities2
        self.valid_entities1 = kgs.valid_entities1
        self.valid_entities2 = kgs.valid_entities2
        self.test_entities1 = kgs.test_entities1
        self.test_entities2 = kgs.test_entities2

        self.attr_value_pairs = kgs.kg1.attr_value_list + kgs.kg2.attr_value_list
        self.value_list = kgs.value_list
        self.primary_values = [val[0] for val in self.value_list]
        self.secondary_values = [val[1:args.select_attr_K] for val in self.value_list]

        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.layer_num = 0 if args.model == 'Only Attr' else 1
        self.activation = tf.nn.leaky_relu

        self.entity_embeddings = entity_embs
        self.value_input = value_embs if args.model != 'Only Rel' else None
        self.attribute_embeddings = attr_embs
        self.relation_embeddings_raw = rel_embs
        self.layers = []
        self.outputs = []

        self.session = tf.Session(config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True),
            allow_soft_placement=True
        ))

        self.learning_rate = args.learning_rate
        self._initialize_variables()
        self._build_training_graph()

        tf.global_variables_initializer().run(session=self.session)
        if self.value_input is not None:
            self.session.run(tf.assign(self.value_embedding_var, self.value_placeholder),
                             {self.value_placeholder: self.value_input})

    def _initialize_variables(self):
        self.entity_padding = tf.constant(0, dtype=tf.float32, shape=(1, self.args.dim))

        if self.value_input is not None:
            self.use_rel = True
            self.no_attr_vector = embed_init(1, 768, "no_attr", method='glorot_uniform_initializer')

            self.value_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.value_num + 1, 768])
            with tf.variable_scope("value_embeddings"):
                self.value_embedding_var = tf.get_variable('value_embedding',
                                                           shape=[self.value_num + 1, 768],
                                                           dtype=tf.float32)

            with tf.variable_scope("relation_embeddings"):
                self.relation_embeddings = tf.Variable(self.relation_embeddings_raw, trainable=True, dtype=tf.float32)
                self.relation_embeddings = tf.concat([
                    tf.nn.l2_normalize(self.relation_embeddings, axis=1),
                    tf.constant(0, dtype=tf.float32, shape=(1, 768))
                ], axis=0)

            with tf.variable_scope("attribute_embeddings"):
                self.attribute_embeddings = tf.Variable(self.attribute_embeddings, trainable=True, dtype=tf.float32)

            with tf.variable_scope("relation_mapping"):
                self.relation_mapping = tf.get_variable("rel_map", shape=[768, self.args.dim],
                                                        dtype=tf.float32,
                                                        initializer=tf.initializers.glorot_normal())

        else:
            with tf.variable_scope("entity_embeddings"):
                self.entity_embeddings = tf.get_variable('entity_embedding',
                                                         shape=[self.entity_num, self.args.dim],
                                                         dtype=tf.float32,
                                                         initializer=tf.initializers.glorot_normal())

            with tf.variable_scope("value_embeddings"):
                value_embeddings = embed_init(self.value_num, self.args.dim, "value_init")
                value_embeddings = tf.nn.l2_normalize(value_embeddings, axis=1)
                self.value_embedding_var = tf.concat([
                    value_embeddings,
                    tf.constant(0, dtype=tf.float32, shape=(1, self.args.dim))
                ], axis=0)

            with tf.variable_scope("relation_embeddings"):
                self.relation_embeddings = tf.get_variable('rel_embedding',
                                                           shape=[self.rel_num, self.args.dim],
                                                           dtype=tf.float32,
                                                           initializer=tf.initializers.glorot_normal())
                self.relation_embeddings = tf.nn.l2_normalize(self.relation_embeddings, axis=1)
                self.relation_embeddings = tf.concat([
                    self.relation_embeddings,
                    tf.constant(0, dtype=tf.float32, shape=(1, self.args.dim))
                ], axis=0)

    def _build_training_graph(self):
        self.loss = 0.0

        self.source_entities = tf.placeholder(tf.int32, shape=[None])
        self.target_entities = tf.placeholder(tf.int32, shape=[None])
        self.ce_labels = tf.placeholder(tf.float32, shape=[None, None])
        self.mi_labels = tf.placeholder(tf.int32, shape=[None])

        if self.args.model != 'Only Rel':
            self._generate_entity_embeddings()

        self.entity_embeddings = tf.concat([self.entity_embeddings, self.entity_padding], axis=0)

        if self.use_rel:
            self.relation_embeddings = tf.matmul(self.relation_embeddings, self.relation_mapping)
            self.relation_embeddings = tf.nn.l2_normalize(self.relation_embeddings, axis=1)

        self.attribute_entity_embeddings = tf.nn.l2_normalize(self.entity_embeddings, axis=1)
        self._graph_convolution()

        self.relation_entity_embeddings = self.outputs[-1]

        self.nce_estimator = NCE_Estimator(temperature=self.args.mi_t, weight=self.args.mi_w)
        attr_input = tf.nn.embedding_lookup(self.attribute_entity_embeddings, self.source_entities)
        rel_input = tf.nn.embedding_lookup(self.relation_entity_embeddings, self.source_entities)

        self.loss += self.nce_estimator.forward(attr_input, self.attribute_entity_embeddings, self.mi_labels)
        self.loss += self.nce_estimator.forward(rel_input, self.relation_entity_embeddings, self.mi_labels)

        combined_embeddings = tf.concat([self.attribute_entity_embeddings, self.relation_entity_embeddings], axis=1)
        combined_embeddings = tf.nn.l2_normalize(combined_embeddings, axis=1)

        input_embeddings = tf.nn.embedding_lookup(combined_embeddings, self.source_entities)
        self.loss += self._cross_entropy_loss(input_embeddings, self.ce_labels, combined_embeddings)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _generate_entity_embeddings(self):
        value_concat = tf.nn.embedding_lookup(self.attribute_embeddings, self.kgs.value_attr_concate)
        value_concat = tf.concat([value_concat, self.no_attr_vector], axis=0)

        combined = tf.concat([value_concat, self.value_embedding_var], axis=1)
        dense = tf.keras.layers.Dense(self.args.dim, use_bias=False)
        self.value_embeddings = tf.nn.l2_normalize(dense(combined), axis=1)

        first_values = tf.nn.embedding_lookup(self.value_embeddings, self.primary_values)
        other_values = tf.nn.embedding_lookup(self.value_embeddings, self.secondary_values)
        mask = tf.cast(tf.not_equal(self.secondary_values, self.value_num), tf.float32)
        mask_exp = tf.expand_dims(mask, axis=-1)

        other_values *= mask_exp
        first_values_tiled = tf.tile(tf.expand_dims(first_values, 1), [1, self.args.select_attr_K - 1, 1])
        fused = other_values + first_values_tiled
        fused_flat = tf.reshape(fused, [-1, self.args.dim])

        attention = tf.keras.layers.Dense(128, activation='relu')(fused_flat)
        attention = tf.keras.layers.Dense(1)(attention)
        attention = tf.reshape(attention, [self.entity_num, self.args.select_attr_K - 1])
        attention_masked = tf.where(mask > 0, attention, tf.fill(tf.shape(attention), -1e15))
        attention = tf.nn.softmax(attention_masked, axis=1)
        attention = tf.expand_dims(attention, axis=-1)

        weighted_sum = tf.reduce_sum(attention * other_values, axis=1)
        self.entity_embeddings = weighted_sum + first_values

        with tf.control_dependencies([
            tf.assert_equal(tf.reduce_any(tf.is_inf(self.entity_embeddings)), False),
            tf.assert_equal(tf.reduce_any(tf.is_nan(self.entity_embeddings)), False)
        ]):
            self.entity_embeddings = tf.identity(self.entity_embeddings)
        self.entity_embeddings = tf.nn.l2_normalize(self.entity_embeddings, axis=1)

    def _graph_convolution(self, evaluation=False):
        self.output = list()
        self.entity_embeddings = tf.nn.l2_normalize(self.entity_embeddings, axis=1)
        self.relation_embeddings = tf.nn.l2_normalize(self.relation_embeddings, axis=1)

        output = self.entity_embeddings
        rels = self.relation_embeddings
        self.outputs.append(output)

        if not evaluation:
            output = tf.nn.dropout(output, rate=self.args.input_drop_rate)

        for layer_idx in range(self.layer_num):
            layer = RGATLayer(self.args.dim, self.args.dim, self.args.dim, self.args.dim,
                              self.relation_adj, self.entity_adj, self.args.drop_rate, layer_idx,
                              self.entity_num, self.args.neighbor_num)
            if not evaluation:
                self.layers.append(layer)
            else:
                layer = self.layers[layer_idx]

            output = tf.nn.l2_normalize(layer.call(output, rels), axis=1)
            self.outputs.append(output)

    def _cross_entropy_loss(self, input_embeds, labels, all_embeds):
        similarity = tf.matmul(input_embeds, tf.transpose(all_embeds))
        logits = tf.sigmoid(similarity)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.reduce_mean(loss)

    def train(self, iteration, epochs_per_iter, args, kgs):
        total_time = 0.0
        for epoch in range(epochs_per_iter):
            steps = math.ceil(len(kgs.train_links) / args.batch_size)
            batch_size = math.ceil(len(self.train_entities1) / steps)
            for step in range(steps):
                links = random.sample(self.train_links, batch_size)
                mi_labels = [link[1] for link in links]
                ce_labels = (args.neg_pro / kgs.entities_num) * np.ones((batch_size, kgs.entities_num + 1))
                for idx, link in enumerate(links):
                    ce_labels[idx][link[1]] += (1 - args.neg_pro)
                    ce_labels[idx][-1] = 0
                e1_list = [l[0] for l in links]
                e2_list = [l[1] for l in links]
                feed_dict = {
                    self.source_entities: e1_list,
                    self.target_entities: e2_list,
                    self.ce_labels: ce_labels,
                    self.mi_labels: mi_labels
                }

                start = time.time()
                result = self.session.run({"loss": self.loss, "train": self.optimizer}, feed_dict=feed_dict)
                duration = round(time.time() - start, 2)
                total_time += duration
                logging.info(f"Iteration {iteration} Epoch {epoch+1}, Step {step+1} — Loss: {result['loss']:.6f}, Time: {duration:.2f}s")

        avg_time = round(total_time / epochs_per_iter, 3)
        logging.info(f"[Iteration {iteration}] Average training time per epoch: {avg_time}s")

    def test(self):
        self._graph_convolution(evaluation=True)
        return self._evaluate_alignment(self.test_entities1, self.test_entities2, tag="test")

    def valid(self):
        self._graph_convolution(evaluation=True)
        return self._evaluate_alignment(self.valid_entities1, self.valid_entities2, tag="valid")

    def _evaluate_alignment(self, entities1, entities2, tag="eval"):
        start = time.time()
        embeds1_all, embeds2_all = [], []

        for out in self.outputs:
            e1 = tf.nn.embedding_lookup(out, entities1)
            e2 = tf.nn.embedding_lookup(out, entities2)
            embeds1_all.append(tf.nn.l2_normalize(e1, 1))
            embeds2_all.append(tf.nn.l2_normalize(e2, 1))

        embeds1 = tf.concat(embeds1_all, axis=1)
        embeds2 = tf.concat(embeds2_all, axis=1)
        embeds1 = tf.nn.l2_normalize(embeds1, axis=1).eval(session=self.session)
        embeds2 = tf.nn.l2_normalize(embeds2, axis=1).eval(session=self.session)

        _, hits1, _, mrr, hits5 = greedy_alignment(embeds1, embeds2,
                                                   self.args.ent_top_k,
                                                   self.args.nums_threads,
                                                   'inner', False, 0, True)
        print(f"{tag.title()} evaluation time: {time.time() - start:.1f}s")
        gc.collect()
        return mrr if tag == "valid" else (mrr, hits1, hits5)

    def save_embeddings(self, output_path="../"):
        self._graph_convolution(evaluation=True)
        outputs = [tf.nn.l2_normalize(o, axis=1) for o in self.outputs]
        all_embeds = tf.concat(outputs, axis=1)
        final_embeds = tf.nn.l2_normalize(all_embeds, axis=1).eval(session=self.session)
        dataset_name = self.args.input.split("/")[-2]
        np.save(f"{output_path}/{dataset_name}_layer{self.layer_num}.npy", final_embeds)