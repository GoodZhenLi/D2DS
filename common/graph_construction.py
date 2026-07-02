import numpy as np
from pymatgen.core.structure import IStructure
from ase.atoms import Atoms
import torch
from pymatgen.io.ase import AseAtomsAdaptor


def graph_construction(z, pos, cell, natoms, radius, device, exclude_self=True):
    natoms = natoms.tolist()
    num_images = len(natoms)
    z = z.cpu().numpy()
    pos = pos.cpu().numpy()
    cell = cell.cpu().numpy()
    num_atoms = 0
    src, dst, cell_offsets, offsets = [], [], [], []
    for img in range(num_images):
        charges = z[num_atoms:natoms[img] + num_atoms]
        positions = pos[num_atoms:natoms[img] + num_atoms]
        struct = AseAtomsAdaptor.get_structure(
            Atoms(positions=positions, numbers=charges, cell=cell[img], pbc=[True, True, False])
        )
        c_index, n_index, cell_offset, n_distance = struct.get_neighbor_list(
            r=radius, numerical_tol=1e-6, exclude_self=exclude_self
        )
        offset = np.matmul(cell_offset, cell[img])
        src.append(n_index + num_atoms)
        dst.append(c_index + num_atoms)
        offsets.append(offset)
        cell_offsets.append(cell_offset)
        num_atoms += natoms[img]
    pos = torch.tensor(pos)
    dst = torch.tensor(np.concatenate(dst))
    src = torch.tensor(np.concatenate(src))
    offsets = torch.tensor(np.concatenate(offsets, axis=0))
    cell_offsets = torch.tensor(np.concatenate(cell_offsets, axis=0))
    vec = pos[dst] - (pos[src] + offsets)
    dist = torch.norm(vec + 1e-8, dim=1, p=2)
    return src.long().to(device), dst.long().to(device), dist.float().to(device), vec.float().to(
        device), offsets.float().to(device), cell_offsets.float().to(device)


def edge_generation(z, pos, cell, radius, exclude_self=True):
    if not isinstance(cell, np.ndarray):
        cell=np.array(cell)
    if cell.shape[0]==1:
        cell=cell[0]

    atoms=Atoms(positions=pos.numpy(), numbers=z.numpy(), cell=cell, pbc=[True, True, False])


    structure =AseAtomsAdaptor.get_structure(atoms)
    c_index, n_index, cell_offset, n_distance = structure.get_neighbor_list(
        r=radius, numerical_tol=1e-6, exclude_self=exclude_self
    )
    offset = torch.tensor(np.matmul(cell_offset, cell)).float()
    dst=torch.tensor(c_index).long()
    src=torch.tensor(n_index).long()
    cell_offset=torch.tensor(cell_offset)

    vec = pos[c_index] - (pos[n_index] + offset)
    dist = torch.norm(vec + 1e-8, dim=1, p=2).float()
    return src,dst,dist,vec,cell_offset,offset
