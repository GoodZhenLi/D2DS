from ase.db import connect
from ase.io import read
import sys
from pymatgen.core import tensors
from matplotlib import pyplot as plt

sys.path.append('../')
from torch_geometric.loader import DataLoader
import pandas as pd
import os
import torch
from torch_geometric.data import Data, Dataset
import numpy as np
import json
from tqdm import tqdm
from common.graph_construction import edge_generation

imp2d = "database.json"
with open(imp2d, 'r') as f:
    raw_data = json.load(f)









def data_splitting(seed, train_ratio, valid_ratio,cutoff):
    datalist = []
    for imp in raw_data:
        y = imp["formation_energy"]

        if abs(y) <10:
            pos = torch.tensor(imp["pos"]).float()
            z = torch.tensor(imp["z"]).long()
            cell = torch.tensor(imp["cell"]).view(1, 3, 3).float()
            add_charge = imp["add_charge"]
            defect_site = imp["defect_site"]
            src, dst, dist, vec, cell_offset, offset = edge_generation(z, pos, cell, cutoff)
            new_index = torch.arange(len(z), device=z.device)
            pristine_z = z[new_index != defect_site]
            pristine_pos=pos[new_index != defect_site]
            pristine_src, pristine_dst, pristine_dist, pristine_vec, pristine_cell_offset, pristine_offset = edge_generation(pristine_z,pristine_pos, cell, cutoff)



            k = {
                "pos": pos,
                "z": z,
                "natoms": len(z),
                "cell": cell,
                "y": y,
                "defect_sites": [[defect_site]],
                "add_charges": add_charge,
                "removed_charges": 0,
                "defect_type": "ads" if imp["defect_type"] == "ads" else "int",
                "src": src.tolist(),
                "dst": dst.tolist(),
                "dist": dist,
                "vec": vec,
                "cell_offsets": cell_offset,
                "offsets": offset,
                "pristine_src":pristine_src.tolist(),
                "pristine_dst":pristine_dst.tolist(),
                "pristine_dist":pristine_dist,
                "pristine_vec":pristine_vec,
                "pristine_cell_offset":pristine_offset,
                "pristine_offset":pristine_offset,
                "pristine_z":pristine_z,
                "pristine_pos":pristine_pos
            }
            datalist.append(k)
    num_data = len(datalist)
    idx = np.arange(num_data)
    np.random.seed(seed)
    np.random.shuffle(idx)
    datalist = np.array(datalist)
    train_list = datalist[idx[:int(train_ratio * num_data)]].tolist()
    valid_list = datalist[idx[int(train_ratio * num_data):int((train_ratio + valid_ratio) * num_data)]].tolist()
    test_list = datalist[idx[int((train_ratio + valid_ratio) * num_data):]].tolist()
    return train_list, valid_list, test_list


class dataset(Dataset):
    def __init__(self, data_list):
        super(dataset, self).__init__()
        self.data_list = data_list

    def get(self, index):
        return Data(**self.data_list[index])

    def len(self):
        return len(self.data_list)


def load_dataset(seed, batch_size, train_ratio, valid_ratio,cutoff):  # datset1,dataset2,dataset3
    print("Loading dataset...")
    train_list, valid_list, test_list = data_splitting(seed, train_ratio, valid_ratio,cutoff)
    print(f"Load {len(train_list)} dataset, {len(valid_list)} dataset, {len(test_list)} dataset")
    train_loader = DataLoader(dataset(train_list), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset(valid_list), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset(test_list), batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    with open("database.json", "r") as F:
        raw_data = json.load(F)
    ads = []
    inters = []
    for imp in raw_data:
        y = imp["formation_energy"]
        defect_type = imp["defect_type"]
        if defect_type == "ads":
            ads.append(y)
        else:
            inters.append(y)

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.kdeplot(ads, label="Ads")
    sns.kdeplot(inters, label="Int")
    plt.legend()
    plt.show()
