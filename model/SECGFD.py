import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import dgl.function as fn
import sympy
import scipy
from dgl.nn import GraphConv, GATConv, SAGEConv
import dgl

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

def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas


class SECGFD(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, graph, d=2, high_order=2):
        super(SECGFD, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(BandConv(hid_dim, hid_dim, self.thetas[i], lin=False))
        for j in range(high_order):
            self.conv.append(HighConv(hid_dim, hid_dim, num_layer=j+1))

        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.linear3 = nn.Linear(hid_dim * len(self.conv), hid_dim)
        self.linear4 = nn.Linear(hid_dim, out_dim)

        self.graph_new = dgl.remove_self_loop(self.g)
        self.GCN = GCN(in_dim, hid_dim, in_dim, self.graph_new).to(self.g.device)
        self.act = nn.ReLU()
        self.d = d

    def forward(self, in_feat):
        h = self.linear1(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0]).to(in_feat.device)
        for conv in self.conv:
            h0 = conv(self.g, h)
            h_final = torch.cat([h_final, h0], -1)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        emb = self.GCN(in_feat)
        return h, emb


class HighConv(nn.Module):
    def __init__(self, hid_dim, out_dim, num_layer=1):
        super().__init__()
        self.num_layer = num_layer
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.act = nn.ReLU()

    def Laplacian(self, graph, in_feat, D_invsqrt):
        """ Operation Feat - Feat * D^-1/2 A D^-1/2 """
        graph.ndata['h'] = in_feat * D_invsqrt
        graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        return in_feat - graph.ndata.pop('h') * D_invsqrt
    
    def forward(self, graph, in_feat):
        D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1)
        feat = in_feat
        for n in range(self.num_layer):
            feat = self.Laplacian(graph, feat, D_invsqrt)
        return feat


class LowConv(nn.Module):
    def __init__(self, hid_dim, out_dim, num_layer=1):
        super().__init__()
        self.num_layer = num_layer
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.act = nn.ReLU()
    
    def MessagePassing(self, graph, in_feat, D_invsqrt):
        """ Operation Feat * D^-1/2 A D^-1/2 """
        graph.ndata['h'] = in_feat * D_invsqrt
        graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        return graph.ndata.pop('h') * D_invsqrt
    
    def forward(self, graph, in_feat):
        D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1)
        feat = in_feat
        for n in range(self.num_layer):
            feat = self.MessagePassing(graph, feat, D_invsqrt)
        return feat



class BandConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(BandConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h