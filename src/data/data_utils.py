import os
import re
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch import Tensor
from torch_geometric.datasets import OGB_MAG,DBLP
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
import pickle
from sklearn.model_selection import KFold
def load_data(args):
    file_path = os.path.join(os.getcwd(), 'data/processed_data/NIH_reporter/nodes')

    if args.dataset=='OGB_MAG':

        dataset = OGB_MAG(root='./data', preprocess='metapath2vec', transform=T.ToUndirected())
        data = dataset[0]
        return data

    elif args.dataset=='DBLP':
        dataset = DBLP(root='./data/DBLP')
        data = dataset[0]
        return data

    elif args.dataset=='NIH_dataset':

        pub_feature = np.load(os.path.join(file_path, 'pub_feature.npy'))
        pi_feature = np.load(os.path.join(file_path, 'pi_feature.npy'))
        pi_y = np.load(os.path.join(file_path, 'pi_y.npy'))
        affiliation_feature = np.load(os.path.join(file_path, 'affiliation_feature.npy'))
        venue_feature = np.load(os.path.join(file_path, 'affiliation_feature.npy'))
        author_feature = np.load(os.path.join(file_path, 'author_feature.npy'))
        edge_pi_author = np.load(os.path.join(file_path, 'edge_pi_author.npy'))
        edge_author_pub = np.load(os.path.join(file_path, 'edge_author_pub.npy'))
        edge_pub_author = np.load(os.path.join(file_path, 'edge_pub_author.npy'))
        edge_pub_citation = np.load(os.path.join(file_path, 'edge_pub_citation.npy'))
        edge_affiliation_author = np.load(os.path.join(file_path, 'edge_affiliation_author.npy'))
        edge_venue_pub = np.load(os.path.join(file_path, 'edge_venue_pub.npy'))

        train_mask, val_mask, test_mask = create_mask(pi_feature)
        data = HeteroData()
        data['pi'].x = torch.from_numpy(pi_feature).to(torch.float)
        data['pi'].y = torch.from_numpy(pi_y).to(torch.long)
        data['pi'].train_mask = train_mask
        data['pi'].val_mask = val_mask
        data['pi'].test_mask = test_mask
        data['author'].x = torch.from_numpy(author_feature).to(torch.float)
        data['pub'].x = torch.from_numpy(pub_feature).to(torch.float)
        data['affiliation'].x = torch.from_numpy(affiliation_feature).to(torch.float)
        data['venue'].x = torch.from_numpy(venue_feature).to(torch.float)
        data["pi"].node_id = torch.arange(len(pi_feature))
        data['author'].node_id = torch.arange(len(author_feature))
        data['pub'].node_id = torch.arange(len(pub_feature))
        data['pi', 'IsMappedTo', 'author'].edge_index = torch.from_numpy(edge_pi_author).to(torch.long)
        data['author', 'rev_IsMappedTo', 'pi'].edge_index = torch.flip(torch.from_numpy(edge_pi_author).to(torch.long), [0])
        data['author', 'writes', 'pub'].edge_index = torch.from_numpy(edge_author_pub).to(torch.long)
        data['pub', 'rev_writes', 'author'].edge_index = torch.from_numpy(edge_pub_author).to(torch.long)
        data['pub', 'cite', 'pub'].edge_index = torch.from_numpy(edge_pub_citation).to(torch.long)
        data['affiliation', 'has', 'author'].edge_index = torch.from_numpy(edge_affiliation_author).to(torch.long)
        data['venue', 'host', 'pub'].edge_index = torch.from_numpy(edge_venue_pub).to(torch.long)

        return data



def data_loader(data,args):
    # central_node = 'pi' if args.dataset == 'NIH_dataset' else 'author' if args.dataset == 'DBLP' else 'paper'
    central_node = data.node_types[0]
    train_loader = NeighborLoader(
        data,
        # Sample 15 neighbors for each node and each edge type for 2 iterations:
        num_neighbors=[30] * args.num_layers,
        # Use a batch size of 128 for sampling training nodes of type "paper":
        batch_size=args.batch_size,
        input_nodes=(central_node, data[central_node].train_mask),
    )
    val_loader = NeighborLoader(
        data,
        # Sample 15 neighbors for each node and each edge type for 2 iterations:
        num_neighbors=[30] * args.num_layers,
        # Use a batch size of 128 for sampling training nodes of type "paper":
        batch_size=args.batch_size,
        input_nodes=(central_node, data[central_node].val_mask),
    )
    test_loader = NeighborLoader(
        data,
        # Sample 15 neighbors for each node and each edge type for 2 iterations:
        num_neighbors=[30] * args.num_layers,
        # Use a batch size of 128 for sampling training nodes of type "paper":
        batch_size=args.batch_size,
        input_nodes=(central_node, data[central_node].test_mask),
    )
    return train_loader,val_loader,test_loader

def create_mask(input):
    num_nodes = len(input)
    train_size = int(0.8 * num_nodes)
    val_size = int(0.1 * num_nodes)
    test_size = num_nodes - train_size - val_size  # 剩下的节点用于测试集

    # 随机打乱所有节点索引
    perm = torch.randperm(num_nodes)

    # 根据比例划分索引
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    # 创建掩码
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 设置对应位置为 True
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def create_10_fold_masks_and_save(x_feature, path):
    """
    Create 10-fold cross-validation masks with the last 10% always set as the test set,
    and save all masks to a file for later use.

    Args:
    - pi_feature: numpy array of node features (used for determining the number of nodes).
    - filename: string, the name of the file to save the masks.

    Saves:
    - A list of dictionaries containing 'train_mask', 'val_mask', and 'test_mask' for each fold.
    """

    num_nodes = len(x_feature)
    indices = np.arange(num_nodes)

    # Split the last 10% as the test set
    split = int(0.4 * num_nodes)
    test_idx = indices[split:]
    remaining_idx = indices[:split]

    # Create test mask
    test_mask = np.zeros(num_nodes, dtype=bool)
    test_mask[test_idx] = True

    # Create 10-fold splits for train/validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    train_val_splits = list(kf.split(remaining_idx))

    # Store all the masks in a list
    all_masks = []

    for fold_idx, (train_idx, val_idx) in enumerate(train_val_splits):
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)

        train_mask[remaining_idx[train_idx]] = True
        val_mask[remaining_idx[val_idx]] = True

        # Append the masks for this fold to the list
        all_masks.append({
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        })

    # Save the masks to a file using pickle
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created.")
    file = os.path.join(path,'masks.pkl')
    with open(file, 'wb') as f:
        pickle.dump(all_masks, f)

    print(f"Masks saved to {file}")


def load_masks(path, fold_idx=0):
    """
    Load the masks for a specific fold from the saved file.

    Args:
    - filename: string, the name of the file to load masks from.
    - fold_idx: integer, specifies which fold's masks to load.

    Returns:
    - A dictionary containing 'train_mask', 'val_mask', and 'test_mask' for the specified fold.
    """
    filename = os.path.join(path,'masks.pkl')
    with open(filename, 'rb') as f:
        all_masks = pickle.load(f)

    return all_masks[fold_idx]