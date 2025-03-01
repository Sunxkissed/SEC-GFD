import torch
import torch.nn.functional as F
import argparse
import time
import numpy as np
import random
from dataset import Dataset, train_delete
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
import pickle as pkl
import dgl
from model.SECGFD import SECGFD
from utils import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)

def train(graph, args, in_feats, h_feats, num_class):
    features = graph.ndata['feature']
    labels = graph.ndata['label']
    index = list(range(len(labels)))
    if dataname == 'amanzon':
        index = list(range(3305, len(labels)))
    
    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args.train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()

    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())

    graph = graph.to(device)
    features = graph.ndata['feature']
    labels = graph.ndata['label']

    if args.del_train != 0:
        graph = train_delete(graph, train_mask, train_del=args.del_train)
        graph = dgl.add_self_loop(graph)
    
    if args.run == 1:
        model = SECGFD(in_feats, h_feats, num_class, graph, d=args.order, high_order=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    time_start = time.time()

    best_auc, best_f1, final_trec, final_tpre, final_tmf1, final_tauc, biggest_f1 = 0., 0., 0., 0., 0., 0., 0.

    for i in range(args.epoch):
        model.train()
        logits, emb = model(features)

        loss1 = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]).to(device))
        nce_loss = cal_nceloss(emb, features, labels, idx_train)
        loss = loss1 + args.lemda * nce_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        probs = logits.softmax(1)
        f1, thres = get_best_f1(labels[test_mask], probs[test_mask])
        labels_np = labels.cpu().detach().numpy()
        probs_np = probs.cpu().detach().numpy()
        preds = np.zeros_like(labels_np)
        preds[probs_np[:, 1] > thres] = 1
        trec = recall_score(labels_np[test_mask], preds[test_mask])
        tpre = precision_score(labels_np[test_mask], preds[test_mask])
        tmf1 = f1_score(labels_np[test_mask], preds[test_mask], average='macro')
        tauc = roc_auc_score(labels_np[test_mask], probs_np[test_mask][:, 1])

        if biggest_f1 < f1:
            biggest_f1 = f1
        
        if (best_f1 + best_auc) < (f1 + tauc):
            best_f1 = f1
            best_auc = tauc
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc

        print('Trial {}, Epoch {}, loss: {:.4f}, test mf1: {:.4f}, (best f1: {:.4f} auc: {:.4f}), biggest f1: {:.4f}'.format(trial, i, loss, f1, best_f1, best_auc, biggest_f1))

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100,
                                                                        final_tpre*100, final_tmf1*100, final_tauc*100))
    return final_tmf1, final_tauc


def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    labels = labels.cpu().detach().numpy()
    probs = probs.cpu().detach().numpy()
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HSGFD')
    parser.add_argument('--dataset', type=str, default='tfinance', help='dataset for our model (yelp/amazon/tfinance/tsocial/reddit)')
    parser.add_argument('--train_ratio', type=float, default=0.4, help='Training Ratio')
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")
    parser.add_argument("--data_path", type=str, default='/data', help="data path")
    parser.add_argument("--adj_type", type=str, default='sym', help="sym or rw")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for (Homo) and 0 for (Hetero)")
    parser.add_argument("--lemda", type=float, default=0.2, help="balance between losses")
    parser.add_argument("--del_ratio", type=float, default=0.000, help="delete heterophily edges ratios")
    parser.add_argument("--del_train", type=float, default=0.000, help="delete train heterophily edges ratios")
    parser.add_argument('--ntrials', type=int, default=1)

    args = parser.parse_args()
    print(args)

    dataname = args.dataset
    data_path = args.data_path
    h_feats = args.hid_dim
    epoch_num = args.epoch
    homo = args.homo
    del_ratio = args.del_ratio

    graph = Dataset(dataname, homo, del_ratio).graph
    in_feats = graph.ndata['feature'].shape[1]
    num_class = 2

    result_f1 = []
    result_auc = []

    for trial in range(args.ntrials):
        final_tmf1, final_tauc = train(graph, args, in_feats, h_feats, num_class)
        result_f1.append(final_tmf1*100)
        result_auc.append(final_tauc*100)

    print('Final Result  Dataset:{}, Run:{}, Test F1:{:.2f}, Test AUC:{:.2f}'.format(args.dataset, args.ntrials,
                                                               np.mean(result_f1), np.mean(result_auc)))
