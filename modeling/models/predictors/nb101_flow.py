from __future__ import print_function, absolute_import

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NB101Flow(nn.Module):
    """
    Compatible to NN regression-based predictor of architecture performance.
    """
    def __init__(self,
                 search_space,
                 gcn_out_dims=(128, 128, 128, 128, 128),
                 hidden_dim=96,
                 node_embedding_dim=48,
                 op_embedding_dim=48,
                 gcn_dropout=0.,
                 gcn_kwargs=None,
                 use_bn=False,
                 use_global_node=False,
                 use_final_only=False,
                 share_op_attention=False,
                 other_node_zero=False,
                 input_op_emb_trainable=False,
                 mlp_hiddens=(200,),
                 mlp_dropout=0.1):
        super(NB101Flow, self).__init__()

        if gcn_kwargs is None:
            gcn_kwargs = {'plus_I': True, 'residual_only': 0}

        self.search_space = search_space
        # configs
        self.op_embedding_dim = op_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        self.hidden_dim = hidden_dim
        self.gcn_out_dims = gcn_out_dims
        self.gcn_dropout = gcn_dropout
        self.use_bn = use_bn
        self.use_final_only = use_final_only
        self.use_global_node = use_global_node
        self.share_op_attention = share_op_attention
        self.input_op_emb_trainable = input_op_emb_trainable
        self.vertices = self.search_space.num_vertices
        self.num_op_choices = self.search_space.num_op_choices
        self.none_op_ind = self.search_space.none_op_ind

        self.input_node_emb = nn.Embedding(1, self.node_embedding_dim)
        # Maybe separate output node?
        self.other_node_emb = nn.Parameter(torch.zeros(1, self.node_embedding_dim),
                                           requires_grad=not other_node_zero)

        # self.middle_node_emb = nn.Parameter(torch.zeros((1, self.embedding_dim)),
        #                                     requires_grad=False)
        # # zero is ok
        # self.output_node_emb = nn.Parameter(torch.zeros((1, self.embedding_dim)),
        #                                     requires_grad=False)

        # the last embedding is the output op emb
        self.input_op_emb = nn.Parameter(torch.zeros(1, self.op_embedding_dim),
                                         requires_grad=self.input_op_emb_trainable)
        self.op_emb = nn.Embedding(self.num_op_choices, self.op_embedding_dim)
        self.output_op_emb = nn.Embedding(1, self.op_embedding_dim)
        if self.use_global_node:
            self.global_op_emb = nn.Parameter(torch.zeros((1, self.op_embedding_dim)))
            self.vertices += 1

        self.x_hidden = nn.Linear(self.node_embedding_dim, self.hidden_dim)

        if self.share_op_attention:
            assert len(np.unique(self.gcn_out_dims)) == 1, \
                "If share op attention, all the gcn-flow layers should have the same dimension"
            self.op_attention = nn.Linear(self.op_embedding_dim, self.gcn_out_dims[0])

        # init graph convolutions
        self.gcns = []
        self.bns = []
        in_dim = self.hidden_dim
        for dim in self.gcn_out_dims:
            self.gcns.append(DenseGraphFlow(
                in_dim, dim, self.op_embedding_dim if not self.share_op_attention else dim,
                has_attention=not self.share_op_attention, **(gcn_kwargs or {})))
            in_dim = dim
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(self.vertices))
        self.gcns = nn.ModuleList(self.gcns)
        if self.use_bn:
            self.bns = nn.ModuleList(self.bns)
        self.num_gcn_layers = len(self.gcns)
        self.embedding_dim = in_dim

        dim = self.embedding_dim
        # construct MLP from embedding to score
        self.mlp = []
        for hidden_size in mlp_hiddens:
            self.mlp.append(
                nn.Sequential(
                    nn.Linear(dim, hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)
                )
            )
            dim = hidden_size
        self.mlp.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*self.mlp)

    def embed_and_transform_arch(self, archs):
        adjs = self.input_op_emb.new([arch[0].T for arch in archs])
        op_inds = self.input_op_emb.new([arch[1] for arch in archs]).long()
        if self.use_global_node:
            tmp_ones = torch.ones((adjs.shape[0], 1, 1), device=adjs.device)
            tmp_cat = torch.cat((
                tmp_ones,
                (op_inds != self.none_op_ind).unsqueeze(1).to(torch.float32),
                tmp_ones), dim=2)  # (batch_size, 1, vertices - 1)  vertices - 3 + 2
            adjs = torch.cat(
                (torch.cat((adjs, tmp_cat), dim=1),
                 torch.zeros((adjs.shape[0], self.vertices, 1), device=adjs.device)), dim=2)
            # (batch_size, vertices, vertices - 1), (batch_size, vertices, 1), (batch_size, vertices, vertices)

        op_embs = self.op_emb(op_inds)  # (batch_size, vertices - 2, op_emb_dim)
        b_size = op_embs.shape[0]
        # the input one should not be relevant
        if self.use_global_node:
            op_embs = torch.cat(
                (self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                 op_embs,
                 self.output_op_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
                 self.global_op_emb.unsqueeze(0).repeat([b_size, 1, 1])),
                dim=1)
            # (batch_size, 1, op_emb_dim), (batch_size, vertices - 3, op_emb_dim)
            # (batch_size, 1, op_emb_dim), (batch_size, 1, op_emb_dim)
            # (batch_size, vertices, op_emb_dim)
        else:
            op_embs = torch.cat(
                (self.input_op_emb.unsqueeze(0).repeat([b_size, 1, 1]),
                 op_embs,
                 self.output_op_emb.weight.unsqueeze(0).repeat([b_size, 1, 1])),
                dim=1)
            # (batch_size, 1, op_emb_dim), (batch_size, vertices - 2, op_emb_dim), (batch_size, 1, op_emb_dim)
            # (batch_size, vertices, op_emb_dim)
        node_embs = torch.cat(
            (self.input_node_emb.weight.unsqueeze(0).repeat([b_size, 1, 1]),
             self.other_node_emb.unsqueeze(0).repeat([b_size, self.vertices - 1, 1])),
            dim=1)
        # (batch_size, vertices, node_emb_dim)

        x = self.x_hidden(node_embs)
        # x: (batch_size, vertices, hid_dim)
        # adjs: (batch_size, vertices, vertices), x: (batch_size, vertices, hid_dim)
        # op_embs: (batch_size, vertices, op_emb_dim), op_inds: (batch_size, vertices - 2)
        return adjs, x, op_embs, op_inds

    def forward(self, archs):
        # adjs: (batch_size, vertices, vertices)
        # x: (batch_size, vertices, hid_dim)
        # op_emb: (batch_size, vertices, emb_dim)
        adjs, x, op_emb, op_inds = self.embed_and_transform_arch(archs)
        if self.share_op_attention:
            op_emb = self.op_attention(op_emb)  # (batch_size, vertices, gcn_out_dims[-1])
        y = x
        for i_layer, gcn in enumerate(self.gcns):
            y = gcn(y, adjs, op_emb)
            if self.use_bn:
                shape_y = y.shape
                y = self.bns[i_layer](y.reshape(shape_y[0], -1, shape_y[-1])).reshape(shape_y)
            if i_layer != self.num_gcn_layers - 1:
                y = F.relu(y)
            y = F.dropout(y, self.gcn_dropout, training=self.training)
        # y: (batch_size, vertices, gcn_out_dims[-1])
        if self.use_final_only:
            # only use the output node's info embedding as the embedding
            y = y[:, -1, :]  # (batch_size, gcn_out_dims[-1])
        else:
            y = y[:, 1:, :]  # do not keep the inputs node embedding  (batch_size, vertices - 1, gcn_out_dims[-1])

            # throw away padded info here
            if self.use_global_node:
                y = torch.cat((
                    y[:, :-2, :] * (op_inds != self.none_op_ind)[:, :, None].to(torch.float32),
                    y[:, -2:, :]), dim=1)
                # (batch_size, vertices - 3, gcn_out_dims[-1])
                # (batch_size, 2, gcn_out_dims[-1])
            else:
                y = torch.cat((
                    y[:, :-1, :] * (op_inds != self.none_op_ind)[:, :, None].to(torch.float32),
                    y[:, -1:, :]), dim=1)
                # (batch_size, vertices - 2, gcn_out_dims[-1])
                # (batch_size, 1, gcn_out_dims[-1])
            # (batch_size, vertices - 1, gcn_out_dims[-1])
            y = torch.mean(y, dim=1)  # average across nodes (bs, god)  (batch_size, gcn_out_dims[-1])

        score = self.mlp(y).squeeze()
        return score


class DenseGraphFlow(nn.Module):

    def __init__(self, in_features, out_features, op_emb_dim,
                 has_attention=True, plus_I=False, normalize=False, bias=True,
                 residual_only=None):
        super(DenseGraphFlow, self).__init__()

        self.plus_I = plus_I
        self.normalize = normalize
        self.in_features = in_features
        self.out_features = out_features
        self.op_emb_dim = op_emb_dim
        self.residual_only = residual_only
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if has_attention:
            self.op_attention = nn.Linear(op_emb_dim, out_features)
        else:
            assert self.op_emb_dim == self.out_features
            self.op_attention = nn.Identity()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj, op_emb):
        if self.plus_I:
            adj_aug = adj + torch.eye(adj.shape[-1], device=adj.device).unsqueeze(0)
            if self.normalize:
                degree_invsqrt = 1. / adj_aug.sum(dim=-1).float().sqrt()
                degree_norm = degree_invsqrt.unsqueeze(2) * degree_invsqrt.unsqueeze(1)
                adj_aug = degree_norm * adj_aug
        else:
            adj_aug = adj
        support = torch.matmul(inputs, self.weight)
        if self.residual_only is None:
            # use residual
            output = torch.sigmoid(self.op_attention(op_emb)) * torch.matmul(adj_aug, support) + support
        else:
            # residual only the first `self.residual_only` nodes
            output = torch.sigmoid(self.op_attention(op_emb)) * torch.matmul(adj_aug, support) + torch.cat(
                (support[:, :self.residual_only, :],
                 torch.zeros([support.shape[0], support.shape[1] - self.residual_only, support.shape[2]], device=support.device)),
                dim=1)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
