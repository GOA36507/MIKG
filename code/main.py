import argparse
import math
import random
import time
import sys
import gc
import numpy as np

import warnings
warnings.filterwarnings('ignore')
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from model import MIKG
from train_funcs import get_model
import logging

logging.basicConfig(
    filename='train_log1.log',  # 日志文件名
    level=logging.INFO,        # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    datefmt='%Y-%m-%d %H:%M:%S'  # 时间格式
)
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='../datasets/D_Y_15K_V1/')
parser.add_argument('--division', type=str, default='721_5fold/1/')
parser.add_argument('--model', type=str, default='MIKG')
parser.add_argument('--gen_embedding', default=True, action='store_true')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--dim', type=int, default=650)
parser.add_argument('--select_attr_K', type=int, default=15)
parser.add_argument('--mi_w', type=float, default=0.01, help='Parameter for MI alignment')
parser.add_argument('--mi_t', type=float, default=0.1, help='Parameter for MI alignment')
parser.add_argument('--values_dim', type=int, default=768)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--neighbor_num', type=int, default=25)
parser.add_argument('--neg_pro', type=float, default=0.96)
parser.add_argument('--ent_top_k', type=list, default=[1, 5, 10, 50])
parser.add_argument('--drop_rate', type=float, default=0)
parser.add_argument('--input_drop_rate', type=float, default=0.2)
parser.add_argument('--nums_threads', type=int, default=16)
parser.add_argument('--epochs', type=int, default=1000)



if __name__ == '__main__':
    args = parser.parse_args()
    kgs, model = get_model(args.input, MIKG, args)

    hits1, old_hits1 = 0.0, 0.0
    best_mrr,best_hit1,best_hit5 = 0.0, 0.0,0.0
    best_iteration = 0
    epochs_each = 20
    total_iteration = args.epochs // epochs_each
    dec_time = 0
    for iteration in range(1, total_iteration + 1):
        print("iteration", iteration)
        model.train(iteration, epochs_each,args, kgs)
        curr_hits1 = model.valid()
        test_mrr,test_hit1,test_hit5 = model.test()
        if test_mrr > best_mrr:
            best_mrr = test_mrr  
            best_hit1 = test_hit1
            best_hit5 = test_hit5
            best_iteration = iteration
        if curr_hits1 > hits1:
            hits1 = curr_hits1
        if curr_hits1 < old_hits1:
            dec_time += 1            
        old_hits1 = curr_hits1
        if dec_time >= 2:
            break
    print(f"The best value is mrr:{best_mrr},hit1:{best_hit1},hit5:{best_hit5}, found at iteration {best_iteration}.")

    


