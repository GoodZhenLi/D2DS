import sys

sys.path.append('../')
from torch_geometric.loader import DataLoader
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
import numpy as np
from common.graph_construction import edge_generation


def data_splitting(train_ratio, valid_ratio, seed, radius=5, splitting="all"):
    train_list, valid_list, test_list = [], [], []
    data = pd.read_csv("AB2Z4_defect.csv")
    if splitting == "all":
        num_data = len(data)
        idx = np.arange(num_data)
        np.random.seed(seed)
        np.random.shuffle(idx)
        train_idx = idx[:int(train_ratio * num_data)]
        val_idx = idx[int(train_ratio * num_data):int((train_ratio + valid_ratio) * num_data)]
        for i in range(len(data)):
            target = data.formation_energy[i]
            if pd.isna(target):
                print(i)
            if target > 15 or target < -10:
                continue
            add_charges = data.add_charges[i]
            removed_charges = data.removed_charges[i]
            defect_sites = data.defect_sites[i]
            charges = np.array(eval(data.charges[i]))
            positions = np.array(eval(data.positions[i]))
            cell = np.array(eval(data.cell[i]))
            pos = torch.tensor(positions).float()
            z = torch.tensor(charges).long()
            charges[defect_sites] = add_charges
            src, dst, dist, vec, cell_offset, offset = edge_generation(z, pos, cell, radius)
            k = {
                "pos": torch.tensor(positions).float(),
                "z": torch.tensor(charges).long(),
                "natoms": len(charges),
                "cell": torch.tensor(cell).view(1, 3, 3).float(),
                "add_charges": int(add_charges),
                "removed_charges": int(removed_charges),
                "defect_sites": [[defect_sites]],
                "y": torch.tensor(target),
                "defect_type": "vac" if add_charges == 0 else "sub",
                "src": src.tolist(),
                "dst": dst.tolist(),
                "dist": dist,
                "vec": vec,
                "cell_offsets": cell_offset,
                "offsets": offset,
            }
            if i in train_idx:
                train_list.append(k)
            elif i in val_idx:
                valid_list.append(k)
            else:
                test_list.append(k)
    if splitting == "compound":
        num_host = np.max(data["host_id"]) + 1
        idx = np.arange(num_host)
        np.random.seed(seed)
        np.random.shuffle(idx)
        train_idx = idx[:int(train_ratio * num_host)]
        val_idx = idx[int(train_ratio * num_host):int((train_ratio + valid_ratio) * num_host)]

        for i in range(len(data)):
            target = data.formation_energy[i]
            if target > 15 or target < -10:
                continue
            add_charges = data.add_charges[i]
            removed_charges = data.removed_charges[i]
            defect_sites = data.defect_sites[i]
            charges = np.array(eval(data.charges[i]))
            positions = np.array(eval(data.positions[i]))
            cell = np.array(eval(data.cell[i]))
            pos = torch.tensor(positions).float()
            z = torch.tensor(charges).long()
            charges[defect_sites] = add_charges
            src, dst, dist, vec, cell_offset, offset = edge_generation(z, pos, cell, radius)
            k = {
                "pos": torch.tensor(positions).float(),
                "z": torch.tensor(charges).long(),
                "natoms": len(charges),
                "cell": torch.tensor(cell).view(1, 3, 3).float(),
                "add_charges": int(add_charges),
                "removed_charges": int(removed_charges),
                "defect_sites": [[defect_sites]],
                "y": torch.tensor(target),
                "defect_type": "vac" if add_charges == 0 else "sub",
                "src": src.tolist(),
                "dst": dst.tolist(),
                "dist": dist,
                "vec": vec,
                "cell_offsets": cell_offset,
                "offsets": offset,
            }
            if data["host_id"][i] in train_idx:
                train_list.append(k)
            elif data["host_id"][i] in val_idx:
                valid_list.append(k)
            else:
                test_list.append(k)
    print(f'{len(train_list)} train set, {len(valid_list)} valid set, {len(test_list)} test set')
    return train_list, valid_list, test_list


class defect_dataset(Dataset):
    def __init__(self, data_list):
        super(defect_dataset, self).__init__()
        self.data_list = data_list

    def get(self, index):
        return Data(**self.data_list[index])

    def len(self):
        return len(self.data_list)


def load_dataset(train_ratio, seed, batch_size, splitting, radius):  # datset1,dataset2,dataset3
    print("Loading dataset...")
    valid_ratio=1-train_ratio
    train_set, valid_set, test_set = data_splitting(train_ratio, valid_ratio, seed, splitting=splitting, radius=radius)
    if train_ratio + valid_ratio < 1:

        train_loader = DataLoader(defect_dataset(train_set), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(defect_dataset(test_set), batch_size=2 * batch_size)
        valid_loader = DataLoader(defect_dataset(valid_set), batch_size=2 * batch_size)
        print("Finish loading dataset...")
        return train_loader, valid_loader, test_loader
    else:
        train_loader = DataLoader(defect_dataset(train_set), batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(defect_dataset(valid_set), batch_size=2 * batch_size)
        print("Finish loading dataset...")
        return train_loader, valid_loader #here valid_loader is test loader actually
