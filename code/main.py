import argparse
import logging
import time
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.python.util import deprecation

from model import MIKG
from train_funcs import get_model

def configure_logging(log_file='train_log1.log'):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_arguments():
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
    return parser.parse_args()

def train_model(args):
    kgs, model = get_model(args.input, MIKG, args)

    best_valid_hit1 = 0.0
    prev_valid_hit1 = 0.0
    early_stop_count = 0

    best_mrr, best_hit1, best_hit5 = 0.0, 0.0, 0.0
    best_iteration = 0

    epochs_per_iteration = 20
    total_iterations = args.epochs // epochs_per_iteration

    for iteration in range(1, total_iterations + 1):
        print(f"Iteration {iteration}")
        model.train(iteration, epochs_per_iteration, args, kgs)

        valid_hit1 = model.valid()
        test_mrr, test_hit1, test_hit5 = model.test()

        if test_mrr > best_mrr:
            best_mrr = test_mrr
            best_hit1 = test_hit1
            best_hit5 = test_hit5
            best_iteration = iteration

        if valid_hit1 > best_valid_hit1:
            best_valid_hit1 = valid_hit1

        if valid_hit1 < prev_valid_hit1:
            early_stop_count += 1
        else:
            early_stop_count = 0  

        prev_valid_hit1 = valid_hit1

        if early_stop_count >= 2:
            print("Early stopping triggered.")
            break

    print(f"\nBest Test Results — MRR: {best_mrr:.4f}, Hit@1: {best_hit1:.4f}, Hit@5: {best_hit5:.4f}, at Iteration {best_iteration}")

def main():
    configure_logging()
    args = parse_arguments()
    train_model(args)

if __name__ == '__main__':
    main()
