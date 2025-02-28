import torch
import numpy as np


def get_adj_from_edges(g):
    n_nodes = g.num_nodes()
    adj = torch.zeros(n_nodes, n_nodes).to(g.device)
    u, v = g.edges()
    u = torch.unsqueeze(u, 1)
    v = torch.unsqueeze(v, 1)
    edges = torch.cat((u, v), dim=1).T
    adj[edges[0], edges[1]] = 1
    return adj, edges, u, v


def cal_nceloss(emb, features, labels, idx_train):
    train_normal_index = torch.where(labels[idx_train] == 0)[0]
    train_anomaly_index = torch.where(labels[idx_train] == 1)[0]
    sim_score = torch.nn.functional.cosine_similarity(features, emb)
    abn = torch.mean(sim_score[train_anomaly_index])
    nor = torch.mean(sim_score[train_normal_index])
    nce_loss = -torch.log(nor / (abn))

    return nce_loss

