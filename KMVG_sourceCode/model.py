#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2022

@author: Monica
"""

from cmath import inf
import datetime
from email import message
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
import numpy
from utils import generate_kg_batch

class Aggregator(nn.Module):
    def __init__(self, dim, dropout):
        super(Aggregator, self).__init__()
        self.in_dim = dim
        self.out_dim = dim
        self.dropout = dropout
        self.message_dropout = nn.Dropout(dropout)
    
    def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_users + n_entities, in_dim)
        A_in:            (n_users + n_entities, n_users + n_entities), torch.sparse.FloatTensor
        """
        side_embeddings = torch.matmul(A_in, ego_embeddings)
        embeddings = self.message_dropout(side_embeddings)
        return embeddings

class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0.0, name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1)) #in,out.in-out.self
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj, mask_item=None): #hidden=inputs
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1] #N=seqs_len

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

        a_input = self.leakyrelu(a_input).view(batch_size, N, N, self.dim)
        e_0 = torch.matmul(a_input, self.a_0).squeeze(-1).view(batch_size,N,N)
        e_1 = torch.matmul(a_input, self.a_1).squeeze(-1).view(batch_size,N,N)
        e_2 = torch.matmul(a_input, self.a_2).squeeze(-1).view(batch_size,N,N)
        e_3 = torch.matmul(a_input, self.a_3).squeeze(-1).view(batch_size,N,N)


        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask) 
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output

class SessionGraph(Module):
    def __init__(self, opt, n_node, n_relation, h_list, t_list, r_list, laplacian_dict, A_in, num):
        super(SessionGraph, self).__init__()
        self.opt = opt
        self.num = num
        self.n_relation = n_relation
        self.n_node = n_node
        self.h_list = h_list
        self.t_list = t_list
        self.r_list = r_list
        self.hidden_size = opt.hiddenSize
        self.hop = opt.hop
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.r_embed = nn.Embedding(self.n_relation, self.hidden_size)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relation, self.hidden_size, self.hidden_size))
        self.pos_embedding = nn.Embedding(200, self.hidden_size)
        self.local_agg = LocalAggregator(self.hidden_size, self.opt.alpha, dropout=0.0) #alpha for the leakyrelu
        self.A_in = nn.Parameter(torch.sparse.FloatTensor(self.n_node, self.n_node))
        if A_in is not None:
            self.A_in.data = A_in.to(torch.device("cuda:0"))
        self.A_in.requires_grad = False
        self.laplacian_dict = laplacian_dict

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.hop):
            self.aggregator_layers.append(Aggregator(dim = self.hidden_size, dropout=self.opt.dropout))
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.hidden_size, self.hidden_size)) #(11)w3
        self.glu1 = nn.Linear(self.hidden_size, self.hidden_size) #w4
        self.glu2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False) #w5
        self.w_2 = nn.Parameter(torch.Tensor(self.hidden_size, 1)) #q2
        self.w_3 = nn.Parameter(torch.Tensor(2*self.hidden_size, self.hidden_size)) #q2
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()
    def reset_parameters(self): 
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if(weight.layout != torch.sparse_coo):
                weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask, node_embeds, alias_inputs):
        mask = mask.float().unsqueeze(-1)
        hsum = torch.sum(node_embeds * mask, -2)
        hsum2 = torch.sigmoid(hsum * hsum)
        h2sum = torch.sigmoid(torch.sum((node_embeds * mask) * (node_embeds * mask), -2))
        hpna = (hsum2 - h2sum) / (2 * torch.sum(mask, 1))

        batch_size = hidden.shape[0]
        lens = hidden.shape[1]

        pos_emb = self.pos_embedding.weight[:lens]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs1 = torch.sum(hidden * mask, -2) / torch.sum(mask, 1) #(12)
        hs = hs1.unsqueeze(-2).repeat(1, lens, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1) #(11)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs)) #(13)
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1) #session representation
        select = select + hpna
        b = self.embedding.weight[1:self.num] 
        scores = torch.matmul(select, b.transpose(1, 0))  
        return scores
    
    def update_attention_batch(self, h_list, t_list, r_idx, ego_embed):
        r_embed = self.r_embed.weight[r_idx]
        W_r = self.trans_M[r_idx]

        h_embed = ego_embed[h_list]
        t_embed = ego_embed[t_list]

        # Equation (4)
        r_mul_h = torch.matmul(h_embed, W_r)
        r_mul_t = torch.matmul(t_embed, W_r)
        v_list = torch.sum(torch.tanh(r_mul_t) * r_mul_h+r_embed, dim=1)
        return v_list

    def forward(self, inputs, A, mask_item):
        ego_embed = trans_to_cuda(self.embedding.weight)
        origin = ego_embed
        all_embed = []
        A_in  = self.A_in

        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = trans_to_cuda(layer(ego_embed, A_in))
            all_embed.append(ego_embed)

            rows = []
            cols = []
            values = []
            for r_idx in self.laplacian_dict.keys():
                index_list = torch.where(self.r_list == r_idx)
                batch_h_list = trans_to_cuda(self.h_list[index_list])
                batch_t_list = trans_to_cuda(self.t_list[index_list])

                batch_v_list = trans_to_cuda(self.update_attention_batch(batch_h_list, batch_t_list, r_idx, ego_embed))
                rows.append(batch_h_list)
                cols.append(batch_t_list)
                values.append(batch_v_list)

            rows = torch.cat(rows)
            cols = torch.cat(cols)
            values = torch.cat(values)

            indices = torch.stack([rows, cols])
            shape = A_in.shape
            A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
            # Equation (5)
            A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
            self.A_in.data = A_in.to(torch.device("cuda:0"))
            A_in = self.A_in
        
        kg_embed = all_embed[0]
        for i in range(1,self.hop):
            kg_embed += all_embed[i]
        kg_embed /= self.hop
        bs = inputs.size()[0]
        sqlen = inputs.size()[1]
        item_embeds = origin[inputs]
        kg_embeds = kg_embed[inputs]
        node_embeds = item_embeds.view((bs, sqlen, -1))
        node_embeds = trans_to_cuda(node_embeds)
        hidden = self.local_agg(node_embeds, A, mask_item)
        kg_embeds = kg_embeds.view((bs, sqlen, -1))
        kg_embeds = trans_to_cuda(kg_embeds)
        kg_embeds = self.local_agg(kg_embeds, A, mask_item)
        hidden = hidden+kg_embeds
        return hidden, node_embeds
        

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
        # return variable
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data, mode):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = numpy.array(A)
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden, node_embeds = model(items, A, mask)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    get2 = lambda i: node_embeds[i][alias_inputs[i]]
    node_embeds = torch.stack([get2(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask, node_embeds, alias_inputs)


def train_test(model, train_data, test_data):
    # model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    # scaler = GradScaler()
    mode = "train"

    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data, mode)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    mode = "test"
    with torch.no_grad():
        mrr_20, hit_20, ndcg_20, mrr_10, hit_10, ndcg_10 = [], [], [], [], [], []
        slices = test_data.generate_batch(model.batch_size)
        for i in slices:
            targets, scores = forward(model, i, test_data, mode)
            sub_scores_20 = scores.topk(20)[1]
            sub_scores_10 = scores.topk(10)[1]
            sub_scores_20 = trans_to_cpu(sub_scores_20).detach().numpy()
            sub_scores_10 = trans_to_cpu(sub_scores_10).detach().numpy()
            for score_20, target, mask in zip(sub_scores_20, targets, test_data.mask):
                hit_20.append(np.isin(target - 1, score_20))
                if len(np.where(score_20 == target - 1)[0]) == 0:
                    mrr_20.append(0)  
                    ndcg_20.append(0)    
                else:
                    mrr_20.append(1 / (np.where(score_20 == target - 1)[0][0] + 1))
                    ndcg_20.append(1 / (math.log2(1+np.where(score_20 == target - 1)[0][0] + 1)))
            for score_10, target, mask in zip(sub_scores_10, targets, test_data.mask):
                hit_10.append(np.isin(target - 1, score_10))
                if len(np.where(score_10 == target - 1)[0]) == 0:
                    mrr_10.append(0)     
                    ndcg_10.append(0)  
                else:
                    mrr_10.append(1 / (np.where(score_10 == target - 1)[0][0] + 1))
                    ndcg_10.append(1 / (math.log2(1+np.where(score_10 == target - 1)[0][0] + 1)))
        hit_20 = np.mean(hit_20)*100
        mrr_20 = np.mean(mrr_20) * 100
        ndcg_20 = np.mean(ndcg_20) * 100
        hit_10 = np.mean(hit_10) * 100
        mrr_10 = np.mean(mrr_10) * 100
        ndcg_10 = np.mean(ndcg_10) * 100
        return mrr_20, hit_20, ndcg_20, mrr_10, hit_10, ndcg_10
