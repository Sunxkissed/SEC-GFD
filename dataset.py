from dgl.data import FraudYelpDataset, FraudAmazonDataset, RedditDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch
from utils import *
import random


class Dataset:
    def __init__(self, name='tfinance', homo=True, del_ratio=0.005):
        self.name = name
        graph = None
        if name == 'tfinance':
            graph, label_dict = load_graphs('/data/workspace/yancheng/AD/Node_AD/data/tfinance')
            graph = graph[0]
            graph.ndata['label'] = graph.ndata['label'].argmax(1)
            if del_ratio != 0:
                graph = random_delete(graph, del_ratio)
                graph = dgl.add_self_loop(graph)


        elif name == 'tsocial':
            graph, label_dict = load_graphs('/data/workspace/yancheng/AD/Node_AD/data/tsocial')
            graph = graph[0]
            if del_ratio != 0:
                graph = random_delete(graph, del_ratio)
                graph = dgl.add_self_loop(graph)
            

        elif name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                # graph = dgl.add_self_loop(graph)
            if del_ratio != 0:
                graph = random_delete(graph, del_ratio)
            if homo:
                graph = dgl.add_self_loop(graph)

        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                # graph = dgl.add_self_loop(graph)
            if del_ratio != 0:
                graph = random_delete(graph, del_ratio)
            if homo:
                graph = dgl.add_self_loop(graph)
        
        elif name == 'reddit':
            dataset = load_graphs('/data/workspace/yancheng/AD/Node_AD/data/reddit')
            graph = dataset[0][0]
            if del_ratio != 0:
                graph = random_delete(graph, del_ratio)
            graph = dgl.add_self_loop(graph)
            
        else:
            print('no such dataset')
            exit(1)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)

        self.graph = graph


def random_delete(graph, del_ratio):
    labels = graph.ndata['label']
    adj, edges, u, v = get_adj_from_edges(graph)
    sum = torch.sum(torch.concat((labels[u], labels[v]), dim=1), dim=1)
    index = torch.nonzero(sum == 1)
    he_edge_num = index.shape[0]    # 异质边条数
    threshold = int(del_ratio * he_edge_num)
    edge_to_move = index[torch.randperm(index.size(0))[:threshold]]
    graph_new = dgl.remove_edges(graph, list(edge_to_move))
    return graph_new


def train_delete(graph, train_mask, train_del):
    labels = graph.ndata['label']
    adj, edges, u, v = get_adj_from_edges(graph)
    false_indices = torch.where(train_mask == False)[0].to(graph.device)
    train_edge = torch.nonzero(torch.isin(v, false_indices))[:, 0]
    sum = torch.sum(torch.concat((labels[u], labels[v]), dim=1), dim=1)
    sum[train_edge] = 0
    index = torch.nonzero(sum == 1)
    he_edge_num = index.shape[0]
    threshold = int(train_del * he_edge_num)
    edge_to_move = index[torch.randperm(index.size(0))[:threshold]]
    graph_new = dgl.remove_edges(graph, list(edge_to_move))
    return graph_new

    
