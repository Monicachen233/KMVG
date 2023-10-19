#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2022

@author: Monica
"""

import argparse
import pickle
import time
from utils import build_graph
from utils import Data, split_validation
from model import *
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='amazon_software', help='dataset name: amazon_software/yelp/cosmetics')
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ') #10
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--dropout', type=float, default=0.7, help='dropout')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hop', type=int, default=2, help='n_hops')
opt = parser.parse_args()
print(opt)


def main():
    os.environ['CUDA_VISIBLE_DEVICE']='0'

    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    kg_file = "../datasets/" + opt.dataset + "/kg.txt"
    A_in, n_entities, n_relations, h_list, t_list, r_list, laplacian_dict = build_graph(kg_file)
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    if opt.dataset == 'amazon_software':
        n_node = 21664
    elif opt.dataset == 'yelp':
        n_node = 27097
    elif opt.dataset == 'cosmetics':
        n_node = 41374
    else :
        n_node = 100

    model = trans_to_cuda(SessionGraph(opt, n_entities, n_relations, h_list, t_list, r_list, laplacian_dict, A_in, n_node))

    start = time.time()
    best_result = [0, 0, 0, 0, 0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0, 0, 0, 0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        mrr_20, hit_20, ndcg_20, mrr_10, hit_10, ndcg_10 = train_test(model, train_data, test_data)
        flag = 0
        if mrr_20 >= best_result[0]:
            best_result[0] = mrr_20
            best_epoch[0] = epoch
            flag = 1
        if hit_20 >= best_result[1]:
            best_result[1] = hit_20
            best_epoch[1] = epoch
            flag = 1
        if ndcg_20 >= best_result[2]:
            best_result[2] = ndcg_20
            best_epoch[2] = epoch
            flag = 1
        if mrr_10 >= best_result[3]:
            best_result[3] = mrr_10
            best_epoch[3] = epoch
            flag = 1
        if hit_10 >= best_result[4]:
            best_result[4] = hit_10
            best_epoch[4] = epoch
            flag = 1
        if ndcg_10 >= best_result[5]:
            best_result[5] = ndcg_10
            best_epoch[5] = epoch
            flag = 1
        print('Best Result:')
        print('\tMMR@20:\t%.4f\tRecall@20:\t%.4f\tNDCG@20:\t%.4f\tMRR@10:\t%.4f\tRecall@10:\t%.4f\tNDCG@10:\t%.4f\tepoch:\t%d,\t%d,\t%d,\t%d,\t%d,\t%d\t'% (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5], best_epoch[0], best_epoch[1], best_epoch[2], best_epoch[3], best_epoch[4], best_epoch[5]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
