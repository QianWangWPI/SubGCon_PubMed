import os
import math
import random
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_global_seeds(seed):
    """
    Set global seed for reproducibility
    """
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass

    np.random.seed(seed)
    random.seed(seed)

def arg_parse():
    parser = argparse.ArgumentParser(description='train.py and calibration.py share the same arguments')
    parser.add_argument('--seed', type=int, default=10, help='Random Seed')
    parser.add_argument('--dataset', type=str, default='DBLP', choices=['NIH_dataset','DBLP','OGB_MAG'])
    parser.add_argument('--init', type=str, default='HETERO', choices=['HETERO'])
    parser.add_argument('--hidden_channels', type=int, default=64, help='hidden channels')
    parser.add_argument('--out_channels', type=int, default=4, help='out channels')
    parser.add_argument('--num_layers', type=int, default=4, help='numbers of layers')
    parser.add_argument('--batch_size', type=int, default=128, help='numbers of layers')
    parser.add_argument('--epochs', type=int, default=50, help='numbers of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for training phase')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate. 1.0 denotes drop all the weights to zero')
    parser.add_argument('--calibration', type=str, default='SubGCon', choices=['SubgCon'])
    parser.add_argument('--n_bins', type=int, default=20, help='ece n_bins')

    args = parser.parse_args()
    return args


def plot_acc_calibration(all_logits, all_labels, n_bins, args):
    output = torch.softmax(all_logits, dim=1)
    pred_label = torch.max(output, 1)[1]
    p_value = torch.max(output, 1)[0]
    ground_truth = all_labels
    confidence_all, confidence_acc = np.zeros(n_bins), np.zeros(n_bins)
    for index, value in enumerate(p_value):
        interval = int(value / (1 / n_bins) - 0.0001)
        confidence_all[interval] += 1
        if pred_label[index] == ground_truth[index]:
            confidence_acc[interval] += 1
    for index, value in enumerate(confidence_acc):
        if confidence_all[index] == 0:
            confidence_acc[index] = 0
        else:
            confidence_acc[index] /= confidence_all[index]

    start = np.around(1 / n_bins / 2, 3)
    step = np.around(1 / n_bins, 3)
    plt.figure(figsize=(6, 4))
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams["font.weight"] = "bold"
    plt.bar(np.around(np.arange(start, 1.0, step), 3), confidence_acc,
            alpha=0.7, width=0.03, color='blue', label='Outputs')
    plt.bar(np.around(np.arange(start, 1.0, step), 3),
            np.around(np.arange(start, 1.0, step), 3), alpha=0.7, width=0.03, color='orange', label='Expected')
    plt.plot([0, 1], [0, 1], ls='--', c='k')
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.tick_params(labelsize=13)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.title(args.dataset, fontsize=16, fontweight="bold")
    plt.legend(fontsize=14)
    save_dir = "./output"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}/{args.dataset}_calibration.png', format='png', dpi=300, pad_inches=0, bbox_inches='tight')
    save_path = os.path.join(save_dir, f"{args.dataset}_logits_labels.txt")
    with open(save_path, 'w') as f:
        for logits, label in zip(all_logits.tolist(), all_labels.tolist()):
            logits_str = ', '.join(map(str, logits))
            f.write(f"{logits_str}\t{label}\n")

def plot_histograms(content_a, content_b, labeltitle, args):
    # Plot histogram of correctly classified and misclassified examples
    global conf_histogram

    plt.figure(figsize=(6, 4))
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams["font.weight"] = "bold"
    sns.distplot(content_a, kde=False, bins=args.n_bins, norm_hist=False, fit=None, label=labeltitle[0])
    sns.distplot(content_b, kde=False, bins=args.n_bins, norm_hist=False,  fit=None, label=labeltitle[1])
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.tick_params(labelsize=13)
    plt.title(args.dataset, fontsize=16, fontweight="bold")
    plt.legend(fontsize=14)
    plt.savefig('output/' + args.dataset +'_histogram.png', format='png', transparent=True, dpi=300,
                pad_inches=0, bbox_inches = 'tight')
    # plt.show()

if __name__ == '__main__':
    args = arg_parse()
    print(args)