import torch
from torch_sparse import SparseTensor


def defect_site_in_batch(defect_site, natoms):
    A=[]
    defect_site_batch =[]
    cum_natoms = torch.cumsum(natoms, dim=0)
    for i in range(len(defect_site)):
        if i > 0:
            A.append(torch.tensor(defect_site[i],device=natoms.device)+cum_natoms[i - 1])

        else:
            A.append(torch.tensor(defect_site[i],device=natoms.device))
        defect_site_batch.append(torch.zeros(len(defect_site[i]))+i)
    return torch.cat(A, dim=0).long(),torch.cat(defect_site_batch, dim=0).long().to(natoms.device)

def triplets(edge_index, cell_offsets, num_nodes: int):
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(
        row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
    )
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()

    # Edge indices (k->j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()
    idx_ji = adj_t_row.storage.row()

    # Remove self-loop triplets d->b->d
    # Check atom as well as cell offset
    cell_offset_kji = cell_offsets[idx_kj] + cell_offsets[idx_ji]
    mask = (idx_i != idx_k) | torch.any(cell_offset_kji != 0, dim=-1)

    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
    idx_kj, idx_ji = idx_kj[mask], idx_ji[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji
def angles(pos, edge_index, cell_offsets, num_nodes, use_pbc, offsets):
    _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
        edge_index,
        cell_offsets,
        num_nodes
    )
    # Calculate angles.
    pos_i = pos[idx_i].detach()
    pos_j = pos[idx_j].detach()
    if use_pbc:
        pos_ji, pos_kj = (
            pos[idx_j].detach() - pos_i + offsets[idx_ji],
            pos[idx_k].detach() - pos_j + offsets[idx_kj],
        )
    else:
        pos_ji, pos_kj = (
            pos[idx_j].detach() - pos_i,
            pos[idx_k].detach() - pos_j,
        )

    a = (pos_ji * pos_kj).sum(dim=-1)
    b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
    angle = torch.atan2(b, a)
    return angle, idx_kj, idx_ji

def edge_selection(src,dst, dist,defect_site,cutoff):
    src_find=defect_site-src.unsqueeze(1)
    dst_find = defect_site - dst.unsqueeze(1)
    mask_src=src_find[:,0]*src_find[:,1]
    mask_dst = dst_find[:, 0] * dst_find[:, 1]
    defect_mask=(mask_src==0)&(mask_dst==0)
    non_defect_mask=dist<cutoff
    return non_defect_mask|defect_mask
