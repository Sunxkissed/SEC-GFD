import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, GATConv, SAGEConv
import dgl.function as fn


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, g):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_dim, hid_dim)
        self.conv2 = GraphConv(hid_dim, out_dim)
        self.act = nn.ReLU()
        self.g = g

    def forward(self, in_feat):
        g = self.g
        h = self.conv1(g, in_feat)
        h = self.act(h)
        h = self.conv2(g, h)
        return h


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, g):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, out_feats, 'mean')
        self.act = nn.ReLU()
        self.g = g

    def forward(self, in_feat):
        g = self.g
        h = self.conv1(g, in_feat)
        h = self.act(h)
        h = self.conv2(g, h)
        return h


class GAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, g, num_heads):
        super(GAT, self).__init__()
        self.heads = num_heads
        self.conv1 = GATConv(in_dim, hid_dim, self.heads)
        self.conv2 = GATConv(hid_dim * self.heads, out_dim, self.heads)
        self.linear = nn.Linear(out_dim * self.heads, out_dim)
        self.act = nn.ReLU()
        self.g = g

    def forward(self, in_feat):
        g = self.g
        h = self.conv1(g, in_feat).flatten(1)
        h = self.act(h)
        h = self.conv2(g, h).flatten(1)
        h = self.linear(h)
        return h


class SGC(nn.Module):
    def __init__(self, nlayers, in_dim, emb_dim, dropout, sparse):
        super(SGC, self).__init__()
        self.dropout = dropout
        self.sparse = sparse
        self.linear = nn.Linear(in_dim, emb_dim)
        self.k = nlayers

    def forward(self, g, x):
        x = torch.relu(self.linear(x))

        if self.sparse:
            with g.local_scope():
                g.ndata['h'] = x
                for _ in range(self.k):
                    g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
                return g.ndata['h']
        else:
            for _ in range(self.k):
                x = torch.matmul(g, x)
            return x