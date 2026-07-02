import torch
from torch import nn
from common.graph_construction import graph_construction


def defect_interaction(z, pos, cell, batch, defect_site, cutoff, cutoff4defect):
    z_d = z[defect_site]
    pos_d = pos[defect_site]
    batch_d = batch[defect_site]
    _, n_defect = torch.unique(batch_d, return_counts=True)

    src, dst, dist, vec, offsets, cell_offsets = graph_construction(z_d, pos_d, cell, natoms=n_defect,
                                                                    radius=cutoff4defect, device=z.device,
                                                                    exclude_self=True)
    mask = dist >= cutoff
    src, dst, dist, vec, offsets, cell_offsets = src[mask], dst[mask], dist[mask], vec[mask], offsets[mask], \
    cell_offsets[mask]
    src_back = defect_site[src]
    dst_back = defect_site[dst]
    return src_back, dst_back, dist, vec, offsets, cell_offsets
