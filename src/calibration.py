# import pandas as pd
import os
import abc
# import re
import numpy as np
# from torch_geometric.data import HeteroData
import torch
from pathlib import Path
import copy
# from torch import Tensor
from train import compute_ece
# import torch_geometric.transforms as T
from data.data_utils import load_data, data_loader

import torch.nn.functional as F
from utils import set_global_seeds, arg_parse,plot_acc_calibration,plot_histograms
from torch import nn, Tensor, LongTensor, BoolTensor
from typing import NamedTuple
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from sklearn.metrics import f1_score, precision_score, recall_score
# from model import homoGNN, HeteroGNN, HeteroGNN2, HeteroGNN3, SubGCon,SubGCon2,TS
from model import homoGNN, NIH, DBLP, OGB_MAG, SubGCon,SubGCon2
from tqdm import tqdm
import json
from tqdm import tqdm
import warnings
import time
import string
warnings.filterwarnings("ignore")



def main(args):
    data = load_data(args)
    central_node = data.node_types[0]
    n_class = len(set(np.array(data[central_node].y)))
    # print(set(np.array(data['author'].y)))

    train_loader, val_loader, test_loader = data_loader(data, args)
    # batch = next(iter(train_loader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset == 'NIH_dataset':
        pretrained_model = NIH(hidden_channels=args.hidden_channels, out_channels=n_class, num_layers=args.num_layers, data=data).to(device)
    elif args.dataset == 'DBLP':
        pretrained_model = DBLP(hidden_channels=args.hidden_channels, out_channels=n_class, num_layers=args.num_layers).to(device)

    model_dir = Path(os.path.join('model', args.dataset))
    file_name = os.path.join(model_dir,'best_model.pt')
    pretrained_model.load_state_dict(torch.load(file_name))

    pretrained_model.eval()
    all_labels = []
    all_logits = []
    central_node = data.node_types[0]
    with torch.no_grad():

        for batch in train_loader:
            batch = batch.to(device)

            batch_size = batch[central_node].batch_size
            logits,_ = pretrained_model(batch.x_dict, batch.edge_index_dict)
            loss = F.cross_entropy(logits[:batch_size], batch[central_node].y[:batch_size])

            true_labels = batch[central_node].y[:batch_size]

            all_labels.append(true_labels.cpu())
            all_logits.append(logits[:batch_size].cpu())

    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)
    ece_loss = compute_ece(all_logits, all_labels, n_bins=15)*100

    print(all_logits.shape)
    print(ece_loss)

    if args.dataset == 'NIH_dataset':
        temp_model = SubGCon(pretrained_model,hidden_channels=args.hidden_channels, out_channels=n_class,
                             num_layers=args.num_layers, data=data).to(device)
    elif args.dataset == 'DBLP':
        temp_model = SubGCon2(pretrained_model, hidden_channels=args.hidden_channels, out_channels=args.out_channels,
                             num_layers=args.num_layers).to(device)
        # temp_model = TS(pretrained_model).to(device)
    #
    #

    optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.01, weight_decay=args.weight_decay)
    for epoch in range(1, 200):
        print(f'Epoch {epoch}/{args.epochs}')

        train_loss = train(temp_model, val_loader, optimizer, device, args)
    val_loss, ece_loss = evaluate(temp_model, train_loader, device, args)
    print(ece_loss*100)
    #
    #
    # temp_model.train()
    # total_examples = total_loss = 0
    # loop = tqdm(enumerate(train_loader), total=len(train_loader))
    # optimizer = torch.optim.Adam(temp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # for i in range(args.epochs):
    #
    #     for index, batch in loop:
    #         optimizer.zero_grad()
    #         batch = batch.to(device)
    #         batch_size = batch[central_node].batch_size
    #         out = temp_model(batch.x_dict, batch.edge_index_dict)
    #         loss = F.cross_entropy(out[:batch_size], batch[central_node].y[:batch_size])
    #         loss.backward()
    #         optimizer.step()
    #
    #         pred = out[:batch_size].argmax(dim=1)
    #         true_labels = batch[central_node].y[:batch_size]
    #         correct = (pred == true_labels).sum().item()
    #         accuracy = correct / batch_size
    #
    #         loop.set_postfix(loss=loss.item(), acc=accuracy)
    #         total_examples += batch_size
    #         total_loss += float(loss) * batch_size
    #
    # all_labels = []
    # all_logits = []
    # all_confidences = []  # 用于存储每个样本的置信度
    # correct_indices = []  # 用于存储分类是否正确的布尔值
    # with torch.no_grad():
    #     temp_model.eval()
    #     for batch in train_loader:
    #         batch = batch.to(device)
    #         batch_size = batch[central_node].batch_size
    #         logits = temp_model(batch.x_dict, batch.edge_index_dict)
    #         loss = F.cross_entropy(logits[:batch_size], batch[central_node].y[:batch_size])
    #
    #         pred = logits[:batch_size].argmax(dim=1)
    #         true_labels = batch[central_node].y[:batch_size]
    #         correct = (pred == true_labels).sum().item()
    #
    #         total_loss += float(loss) * batch_size
    #         total_examples += batch_size
    #         confidence = torch.softmax(logits[:batch_size], dim=1).cpu()
    #         confidence = torch.max(confidence, 1)[0]  # 获取每个样本的最大置信度值
    #         correct_index = (true_labels == pred).cpu()
    #         all_labels.append(true_labels.cpu())
    #         all_logits.append(logits[:batch_size].cpu())
    #         all_confidences.append(confidence)
    #         correct_indices.append(correct_index)
    #
    #
    # all_labels = torch.cat(all_labels)
    # all_logits = torch.cat(all_logits)
    # all_confidences = torch.cat(all_confidences)
    # correct_indices = torch.cat(correct_indices)
    #
    # plot_acc_calibration(all_logits, all_labels, args.n_bins, args)
    # plot_histograms(
    #     all_confidences[correct_indices],
    #     all_confidences[~correct_indices],
    #     ['Correct', 'Incorrect'],
    #     args)
    # ece_loss = compute_ece(all_logits, all_labels, n_bins=args.n_bins)
    # print(ece_loss)

def train(model, train_loader, optimizer, device, args):
    central_node = 'pi' if args.dataset == 'NIH_dataset' else 'author' if args.dataset == 'DBLP' else 'paper'
    model.train()
    total_examples = total_loss = 0
    loop = tqdm(enumerate(train_loader), total=len(train_loader))

    for index, batch in loop:
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch[central_node].batch_size

        out,_ = model(batch.x_dict, batch.edge_index_dict)

        loss = F.cross_entropy(out[:batch_size], batch[central_node].y[:batch_size])

        loss_ece = torch.tensor(compute_ece(out[:batch_size], batch[central_node].y[:batch_size], n_bins=15))
        loss = loss+loss_ece
        loss.backward()
        optimizer.step()

        pred = out[:batch_size].argmax(dim=1)
        true_labels = batch[central_node].y[:batch_size]
        correct = (pred == true_labels).sum().item()
        accuracy = correct / batch_size

        loop.set_postfix(loss=loss.item(), acc=accuracy)
        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples

def evaluate(model, loader, device, args):
    central_node = 'pi' if args.dataset == 'NIH_dataset' else 'author'
    model.eval()
    total_examples = total_loss = total_correct = 0
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_size = batch[central_node].batch_size
            logits,_ = model(batch.x_dict, batch.edge_index_dict)
            loss = F.cross_entropy(logits[:batch_size], batch[central_node].y[:batch_size])

            pred = logits[:batch_size].argmax(dim=1)
            true_labels = batch[central_node].y[:batch_size]
            correct = (pred == true_labels).sum().item()

            total_loss += float(loss) * batch_size
            total_correct += correct
            total_examples += batch_size

            all_preds.append(pred.cpu())
            all_labels.append(true_labels.cpu())
            all_logits.append(logits[:batch_size].cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)

    accuracy = total_correct / total_examples
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    ece_loss = compute_ece(all_logits, all_labels, n_bins=15)

    print(f'Loss: {total_loss / total_examples:.4f}, Accuracy: {accuracy:.4f}, '
          f'F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, '
          f'ECE: {ece_loss:.4f}')
    return total_loss / total_examples, ece_loss

if __name__ == '__main__':
    args = arg_parse()
    set_global_seeds(args.seed)
    main(args)