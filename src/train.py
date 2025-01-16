# import pandas as pd
import os
# import re
import numpy as np
# from torch_geometric.data import HeteroData
import torch
from pathlib import Path
import copy
# from torch import Tensor
# import torch_geometric.transforms as T
from data.data_utils import load_data, data_loader, create_10_fold_masks_and_save,load_masks
from sklearn.model_selection import KFold
import torch.nn.functional as F
from utils import set_global_seeds, arg_parse

from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from model import homoGNN, NIH, DBLP, OGB_MAG
from tqdm import tqdm
import json
from tqdm import tqdm
import warnings
import time
import string
warnings.filterwarnings("ignore")



def main(args):
    # central_node = 'pi' if args.dataset == 'NIH_dataset' else 'author' if args.dataset == 'DBLP' else 'paper'
    file_path = os.path.join(os.getcwd(), 'data\\{}'.format(args.dataset))

    # file_path = os.path.join(dataset_path, 'nodes')
    # data = load_data(file_path, sample=True,und=True)
    data = load_data(args)
    print(data)
    central_node = data.node_types[0]
    n_class = len(set(np.array(data[central_node].y)))

    # create_10_fold_masks_and_save(data[central_node].x,file_path)

    # train_loader, val_loader, test_loader = data_loader(data,args)




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset=='NIH_dataset':
        model = NIH(hidden_channels=args.hidden_channels, out_channels=n_class, num_layers=args.num_layers, data=data).to(device)
    elif args.dataset=='DBLP':
        model = DBLP(hidden_channels=args.hidden_channels, out_channels=n_class, num_layers=args.num_layers).to(device)
    else:
        model = OGB_MAG(hidden_channels=args.hidden_channels, out_channels=n_class, num_heads=2, num_layers=args.num_layers,data=data).to(device)

    # #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # # for i in range(10):
    # all_masks = load_masks(file_path, fold_idx=0)
    # data[central_node].train_mask = all_masks['train_mask']
    # data[central_node].val_mask = all_masks['val_mask']
    # data[central_node].test_mask = all_masks['test_mask']
    #
    # assert not np.any(data[central_node].train_mask & data[central_node].val_mask), "Error: Train and validation masks have same True values at some positions."
    train_loader, val_loader, test_loader = data_loader(data,args)
    run_training(model, train_loader, val_loader, test_loader, optimizer, device, args)
    # loss = train(model, train_loader, optimizer)
    # print(loss)


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
    return total_loss / total_examples, accuracy


def run_training(model, train_loader, val_loader, test_loader, optimizer, device, args):
    patience = 100
    best_val_loss = float('inf')
    state_dict_early_model = None
    curr_step = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}')

        train_loss = train(model, train_loader, optimizer, device, args)
        val_loss, val_acc = evaluate(model, val_loader, device, args)
        test_loss, test_acc = evaluate(model, test_loader, device, args)

        print(f'Epoch {epoch} | Train Loss: {train_loss:.4f}')
        print(f'Epoch {epoch} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        print(f'Epoch {epoch} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict_early_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            curr_step = 0
        else:
            curr_step += 1
            if curr_step >= patience:
                print(f'Early stopping at epoch {epoch}. Best epoch: {best_epoch}')
                break

    # Save the best model
    save_model(state_dict_early_model, args)


# def run_training_10fold(model_class, dataset, args):
#     # 定义10-fold交叉验证
#     kf = KFold(n_splits=10, shuffle=True, random_state=42)
#
#     all_fold_val_accuracies = []
#     all_fold_test_accuracies = []
#
#     # 对10个折叠进行遍历
#     for fold, (train_idx, val_idx) in enumerate(kf.split(dataset['pi'].x)):
#         print(f'Fold {fold + 1}')
#
#         # 划分训练、验证和测试集
#         train_mask = torch.zeros(len(dataset['pi'].x), dtype=torch.bool)
#         val_mask = torch.zeros(len(dataset['pi'].x), dtype=torch.bool)
#         test_mask = dataset['pi'].test_mask  # 假设test_mask已经定义好并保持不变
#
#         train_mask[train_idx] = True
#         val_mask[val_idx] = True
#
#         # 更新数据中的mask
#         dataset['pi'].train_mask = train_mask
#         dataset['pi'].val_mask = val_mask
#
#         # 初始化模型和优化器
#         model = model_class()  # 根据传入的模型类创建新的模型实例
#         model.to(args.device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
#
#         patience = 100
#         best_val_loss = float('inf')
#         state_dict_early_model = None
#         curr_step = 0
#         best_epoch = 0
#
#         # 数据加载器的创建（根据你的数据定义可能需要调整）
#         train_loader = create_data_loader(dataset, 'train')
#         val_loader = create_data_loader(dataset, 'val')
#         test_loader = create_data_loader(dataset, 'test')
#
#         # 开始训练和验证
#         for epoch in range(1, args.epochs + 1):
#             print(f'Fold {fold + 1} | Epoch {epoch}/{args.epochs}')
#
#             train_loss = train(model, train_loader, optimizer, args.device, args)
#             val_loss, val_acc = evaluate(model, val_loader, args.device, args)
#             test_loss, test_acc = evaluate(model, test_loader, args.device, args)
#
#             print(f'Epoch {epoch} | Train Loss: {train_loss:.4f}')
#             print(f'Epoch {epoch} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
#             print(f'Epoch {epoch} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')
#
#             # Early stopping check
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 state_dict_early_model = copy.deepcopy(model.state_dict())
#                 best_epoch = epoch
#                 curr_step = 0
#             else:
#                 curr_step += 1
#                 if curr_step >= patience:
#                     print(f'Early stopping at epoch {epoch}. Best epoch: {best_epoch}')
#                     break
#
#         # 保存最佳模型的状态字典
#         if state_dict_early_model is not None:
#             model.load_state_dict(state_dict_early_model)
#
#         # 计算并记录每个折叠的验证和测试准确率
#         _, best_val_acc = evaluate(model, val_loader, args.device, args)
#         _, best_test_acc = evaluate(model, test_loader, args.device, args)
#
#         all_fold_val_accuracies.append(best_val_acc)
#         all_fold_test_accuracies.append(best_test_acc)
#
#         print(f'Fold {fold + 1} | Best Val Acc: {best_val_acc:.4f} | Best Test Acc: {best_test_acc:.4f}')
#
#     # 打印10-fold的平均验证和测试准确率
#     avg_val_acc = sum(all_fold_val_accuracies) / len(all_fold_val_accuracies)
#     avg_test_acc = sum(all_fold_test_accuracies) / len(all_fold_test_accuracies)
#     print(f'10-Fold Cross-Validation | Avg Val Acc: {avg_val_acc:.4f} | Avg Test Acc: {avg_test_acc:.4f}')


# def compute_ece(logits, labels, n_bins=15):
#     """
#     Compute Expected Calibration Error (ECE)
#
#     Parameters:
#     - logits: Model output before applying softmax (torch.Tensor)
#     - labels: Ground truth labels (torch.Tensor)
#     - n_bins: Number of bins to calculate ECE (default: 15)
#
#     Returns:
#     - ece: Computed ECE value
#     """
#     # Apply softmax to convert logits to probabilities
#     probs = F.softmax(logits, dim=1)
#
#     # Get the predicted class probabilities and the predicted classes
#     preds = torch.argmax(probs, dim=1)
#     confidences = torch.max(probs, dim=1)[0]  # Confidence level of the predicted class
#
#     # Create empty arrays to hold bin accuracies and confidences
#     bin_boundaries = torch.linspace(0, 1, n_bins + 1)
#     bin_lowers = bin_boundaries[:-1]
#     bin_uppers = bin_boundaries[1:]
#
#     ece = torch.zeros(1, device=logits.device)
#
#     for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
#         # Get indices of predictions that fall into this bin
#         in_bin = (confidences > bin_lower.item()) * (confidences <= bin_upper.item())
#         prop_in_bin = in_bin.float().mean()  # Proportion of samples in this bin
#
#         if prop_in_bin > 0:
#             # Accuracy in the bin
#             accuracy_in_bin = labels[in_bin] == preds[in_bin]
#             avg_accuracy_in_bin = accuracy_in_bin.float().mean()
#
#             # Average confidence in the bin
#             avg_confidence_in_bin = confidences[in_bin].mean()
#
#             # Calculate ECE contribution for this bin
#             ece += torch.abs(avg_confidence_in_bin - avg_accuracy_in_bin) * prop_in_bin
#
#     return ece.item()



def compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE) using equal-width binning.

    Parameters:
    - logits: Model output before applying softmax (torch.Tensor)
    - labels: Ground truth labels (torch.Tensor)
    - n_bins: Number of bins to calculate ECE (default: 15)

    Returns:
    - ece: Computed ECE value
    """
    # Apply softmax to convert logits to probabilities
    probs = F.softmax(logits, dim=1)

    # Get the predicted classes and their confidence levels
    preds = torch.argmax(probs, dim=1)
    confidences = torch.max(probs, dim=1)[0]

    # Convert labels to the same device as logits and match the types
    corrects = (preds == labels).float()

    # Sort confidences and corresponding corrects
    sorted_confs, sort_indices = torch.sort(confidences)
    sorted_corrects = corrects[sort_indices]

    # Bin the confidences into equal-width bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Initialize variables for tracking calibration
    ece = torch.zeros(1, device=logits.device)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get indices of predictions that fall into this bin
        in_bin = (sorted_confs > bin_lower.item()) * (sorted_confs <= bin_upper.item())
        prop_in_bin = in_bin.float().mean()  # Proportion of samples in this bin

        if prop_in_bin > 0:
            # Accuracy in the bin
            accuracy_in_bin = sorted_corrects[in_bin].float().mean()
            avg_confidence_in_bin = sorted_confs[in_bin].mean()

            # Calculate ECE contribution for this bin
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()





def save_model(state_dict, args):
    # 保存模型到指定目录
    model_dir = Path(os.path.join('model', args.dataset))
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'best_model.pt'
    torch.save(state_dict, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    args = arg_parse()
    set_global_seeds(args.seed)
    # dataset_path = os.path.join(os.getcwd(), 'data/processed_data/NIH_reporter')
    main(args)