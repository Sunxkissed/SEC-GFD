import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

EOS = 1e-10

def get_adj_from_edges(g):
    n_nodes = g.num_nodes()
    adj = torch.zeros(n_nodes, n_nodes).to(g.device)
    u, v = g.edges()
    u = torch.unsqueeze(u, 1)
    v = torch.unsqueeze(v, 1)
    edges = torch.cat((u, v), dim=1).T
    adj[edges[0], edges[1]] = 1
    return adj, edges, u, v 


class High_filter(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, g):
        super(High_filter, self).__init__()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, out_dim)
        self.act = nn.ReLU()
        self.sparse = False
        self.g = g
        self.n_nodes = g.num_nodes()
    
    def normalize_adj(self, adj, mode, sparse):
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]    # D^(-1/2) * A * D^(-1/2)
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    
    def cal_normalized_A(self, g):
        n_nodes = self.n_nodes
        adj, edges, u, v = get_adj_from_edges(g)
        adj += torch.eye(n_nodes).to(g.device)
        nA = self.normalize_adj(adj, 'sym', self.sparse)
        return nA

    def forward(self, in_feat):
        nA = self.cal_normalized_A(self.g)
        Lapla = torch.eye(self.n_nodes) - nA
        x = torch.matmul(Lapla, in_feat)
        h = self.linear1(x)
        h = self.act(h)
        h = torch.matmul(Lapla, h)
        h = self.linear2(h)
        return h


class Lapla_filter(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, graph):
        super().__init__()
        self.g = graph
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.act = nn.ReLU()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, out_dim)
    
    def Laplacian(self, in_feat, D_invsqrt):
        """ Operation Feat * D^-1/2 A D^-1/2 """
        graph = self.g
        graph.ndata['h'] = in_feat * D_invsqrt
        graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        return in_feat - graph.ndata.pop('h') * D_invsqrt
    
    def forward(self, in_feat):
        D_invsqrt = torch.pow(self.g.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1)
        feat = self.linear1(in_feat)
        feat = self.act(feat)
        feat = self.Laplacian(feat, D_invsqrt)
        h = self.linear2(feat)
        return h







