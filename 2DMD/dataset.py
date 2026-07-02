from torch_geometric.loader import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Dataset
import numpy as np
from common.graph_construction import edge_generation


def dataset_splitting(data_path, seed, fold):
    trainset_df = []
    testset_df = []
    validset_df = []
    datasets_csv_list = os.listdir(data_path)
    for csv in datasets_csv_list:
        if ".csv" not in csv:
            continue
        data = pd.read_csv(os.path.join(data_path, csv))
        defect_type = csv.split(".")[0]
        data["defect_type"] = [defect_type for _ in range(len(data['removed_charges']))]
        num_data = len(data)
        idx = np.arange(num_data)
        np.random.seed(seed)
        np.random.shuffle(idx)
        test_idx = idx[int(fold * num_data):int((fold + 0.1) * num_data)]
        valid_idx = idx[int((fold + 0.1)*num_data):int((fold + 0.2) * num_data)]
        train_idx = np.delete(idx, np.arange(int(fold * num_data), int((fold + 0.2) * num_data)))
        test_df = data.loc[test_idx]
        train_df = data.loc[train_idx]
        valid_df = data.loc[valid_idx]
        trainset_df.append(train_df)
        validset_df.append(valid_df)
        testset_df.append(test_df)
    return pd.concat(trainset_df).reset_index(),pd.concat(validset_df).reset_index(),pd.concat(testset_df).reset_index()
def defect_interaction(z, pos, cell, defect_site, cutoff, cutoff4defect):
    z_d = z[defect_site]
    pos_d = pos[defect_site]

    src, dst, dist, vec, offsets, cell_offsets = edge_generation(z_d, pos_d, cell, cutoff4defect)
    mask = dist >= cutoff
    src, dst, dist, vec, offsets, cell_offsets = src[mask], dst[mask], dist[mask], vec[mask], offsets[mask], \
        cell_offsets[mask]
    src_back = defect_site[src]
    dst_back = defect_site[dst]
    return src_back, dst_back, dist, vec, offsets, cell_offsets


def dataset_construction(dataset, target, radius):
    graph_list = []
    for i in tqdm(range(len(dataset))):
        if dataset.iloc[i].isnull().any():
            continue
        else:
            z = torch.tensor(eval(dataset.charges.iloc[i]))
            pos = torch.tensor(eval(dataset.positions.iloc[i]))
            cell = torch.tensor(eval(dataset.cell.iloc[i])).view(1, 3, 3)
            src, dst, dist, vec, cell_offset, offset = edge_generation(z, pos, cell, radius)
            defect_sites=eval(dataset.defect_site.iloc[i])
            k = {
                "z": z,
                "natoms": len(z),
                "pos": pos,
                "cell": cell,
                "src": src.tolist(),
                "dst": dst.tolist(),
                "dist": dist,
                "vec": vec,
                "cell_offsets": cell_offset,
                "offsets": offset,
                "add_charges": torch.tensor(eval(dataset.add_charges.iloc[i])) if type(
                    dataset.add_charges.iloc[i]) == str else dataset.add_charges.iloc[i],
                "removed_charges": torch.tensor(eval(dataset.removed_charges.iloc[i])) if type(
                    dataset.removed_charges.iloc[i]) == str else dataset.removed_charges.iloc[i],
                "defect_sites": defect_sites,
                "num_defect_sites":len(defect_sites),
                "y": torch.tensor(dataset[target].iloc[i]),
                "defect_type": dataset.defect_type.iloc[i] if "defect_type" in dataset.columns else None,
                "weights": torch.tensor(dataset.loss_weight.iloc[i]) if "loss_weight" in dataset.columns else None
            }
            graph_list.append(k)
    return graph_list


class defect_dataset(Dataset):
    def __init__(self, data_list):
        super(defect_dataset, self).__init__()
        self.data_list = data_list

    def get(self, index):
        return Data(**self.data_list[index])

    def len(self):
        return len(self.data_list)


def load_dataset(dataset, seed, batch_size, target, cutoff,
                 fold):
    print("Loading dataset...")
    train_set_csv, valid_set_csv, test_set_csv = dataset_splitting(dataset,seed, fold)
    train_list = dataset_construction(train_set_csv, target, radius=cutoff)
    val_list = dataset_construction(valid_set_csv, target, radius=cutoff)
    test_list = dataset_construction(test_set_csv, target, radius=cutoff)
    test_list += val_list
    train_set = defect_dataset(train_list)
    valid_set = defect_dataset(val_list)
    test_set = defect_dataset(test_list)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    print("Finish loading dataset...")
    return train_loader, valid_loader, test_loader
