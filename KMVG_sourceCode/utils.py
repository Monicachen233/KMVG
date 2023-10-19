#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2022

@author: Monica
"""
import torch
import pandas as pd
import ast
from re import I, T
import networkx as nx
import numpy as np
import pickle
import math
import scipy.sparse as sp
from tqdm import tqdm
from operator import itemgetter
from collections import defaultdict
from tkinter import _flatten
import collections
import random

def symmetric_norm_lap(adj):
    rowsum = np.array(adj.sum(axis=1))

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return norm_adj.tocoo()

def random_walk_norm_lap(adj):
    rowsum = np.array(adj.sum(axis=1))

    d_inv = np.power(rowsum, -1.0).flatten()
    d_inv[np.isinf(d_inv)] = 0
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    return norm_adj.tocoo()

def convert_coo2tensor(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def build_graph(kg_file):
    """build konwledge graph"""
    # load kg_file
    kg_data = pd.read_csv(kg_file, sep='\t', names=['h', 'r', 't'], engine='python')
    kg_data = kg_data.drop_duplicates()
    
    n_relations = max(kg_data['r']) #relation begin from 1
    inverse_kg_data = kg_data.copy()
    inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'})
    inverse_kg_data['r'] += n_relations
    kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True)
    n_relations = max(kg_data['r']) + 1
    n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1

    h_list = []
    t_list = []
    r_list = []

    train_kg_dict = collections.defaultdict(list)
    train_relation_dict = collections.defaultdict(list)

    for row in kg_data.iterrows():
        h, r, t = row[1]
        h_list.append(h)
        t_list.append(t)
        r_list.append(r)

        train_kg_dict[h].append((t, r))
        train_relation_dict[r].append((h, t))

    h_list = torch.LongTensor(h_list)
    t_list = torch.LongTensor(t_list)
    r_list = torch.LongTensor(r_list)

    adjacency_dict = {}
    for r, ht_list in train_relation_dict.items():
        rows = [e[0] for e in ht_list]
        cols = [e[1] for e in ht_list]
        vals = [1] * len(rows)
        adj = sp.coo_matrix((vals, (rows, cols)), shape=(n_entities, n_entities))
        adjacency_dict[r] = adj
    
    laplacian_dict = {}
    for r, adj in adjacency_dict.items():
        laplacian_dict[r] = symmetric_norm_lap(adj)

    A_in = sum(laplacian_dict.values())
    A_in = convert_coo2tensor(A_in.tocoo())

    return A_in, n_entities, n_relations, h_list, t_list, r_list, laplacian_dict

def generate_kg_batch(kg_dict, batch_size,highest_neg_idx):
    exist_heads = kg_dict.keys()
    if batch_size <= len(exist_heads):
        batch_head = random.sample(exist_heads, batch_size)
    else:
        batch_head = [random.choice(exist_heads) for _ in range(batch_size)]

    batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
    for h in batch_head:
        relation, pos_tail = sample_pos_triples_for_h(kg_dict, h, 1)
        batch_relation += relation
        batch_pos_tail += pos_tail

        neg_tail = sample_neg_triples_for_h(kg_dict, h, relation[0], 1, highest_neg_idx)
        batch_neg_tail += neg_tail

    batch_head = torch.LongTensor(batch_head)
    batch_relation = torch.LongTensor(batch_relation)
    batch_pos_tail = torch.LongTensor(batch_pos_tail)
    batch_neg_tail = torch.LongTensor(batch_neg_tail)
    return batch_head, batch_relation, batch_pos_tail, batch_neg_tail

def sample_pos_triples_for_h(kg_dict, head, n_sample_pos_triples):
    pos_triples = kg_dict[head]
    n_pos_triples = len(pos_triples)

    sample_relations, sample_pos_tails = [], []
    while True:
        if len(sample_relations) == n_sample_pos_triples:
            break

        pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
        tail = pos_triples[pos_triple_idx][0]
        relation = pos_triples[pos_triple_idx][1]

        if relation not in sample_relations and tail not in sample_pos_tails:
            sample_relations.append(relation)
            sample_pos_tails.append(tail)
    return sample_relations, sample_pos_tails


def sample_neg_triples_for_h(kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
    pos_triples = kg_dict[head]

    sample_neg_tails = []
    while True:
        if len(sample_neg_tails) == n_sample_neg_triples:
            break

        tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
        if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
            sample_neg_tails.append(tail)
    return sample_neg_tails


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. 
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) 
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    # us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    # us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    us_pois = [list(reversed(upois)) + [0] * (len_max - le) if le < len_max else list(reversed(upois[-len_max:]))
               for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) if le < len_max else [1] * len_max
               for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0]) #begin from 1
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        targets = list(map(int, targets))
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            temp = node.tolist() + (max_n_node - len(node)) * [0]
            temp = list(map(int, temp))
            items.append(temp)
            u_A = np.zeros((max_n_node, max_n_node))

            for i in np.arange(len(u_input) - 1):
                u = np.where(node == u_input[i])[0][0]
                u_A[u][u] = 1
                if u_input[i + 1] == 0:
                    break
                v = np.where(node == u_input[i + 1])[0][0]
                if u == v or u_A[u][v] == 4:
                    continue
                u_A[v][v] = 1
                if u_A[v][u] == 2:
                    u_A[u][v] = 4
                    u_A[v][u] = 4
                else:
                    u_A[u][v] = 2
                    u_A[v][u] = 3
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
    
        return alias_inputs, A, items, mask, targets
