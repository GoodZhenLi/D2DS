import torch
# from e3nn.o3 import Irreps
from torch import nn
from torch_sparse import SparseTensor
import torch.nn.functional as F


# class EquivariantLayerNormFast(nn.Module):
#
#     def __init__(self, irreps, eps=1e-5, affine=True, normalization='component'):
#         super().__init__()
#
#         self.irreps = Irreps(irreps)
#         self.eps = eps
#         self.affine = affine
#
#         num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
#         num_features = self.irreps.num_irreps
#
#         if affine:
#             self.affine_weight = nn.Parameter(torch.ones(num_features))
#             self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
#         else:
#             self.register_parameter('affine_weight', None)
#             self.register_parameter('affine_bias', None)
#
#         assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
#         self.normalization = normalization
#
#     def __repr__(self):
#         return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"
#
#     def forward(self, node_input, **kwargs):
#         '''
#             Use torch layer norm for scalar features.
#         '''
#
#         dim = node_input.shape[-1]
#
#         fields = []
#         ix = 0
#         iw = 0
#         ib = 0
#
#         for mul, ir in self.irreps:  # mul is the multiplicity (number of copies) of some irrep type (ir)
#             d = ir.dim
#             field = node_input.narrow(1, ix, mul * d)
#             ix += mul * d
#
#             if ir.l == 0 and ir.p == 1:
#                 weight = self.affine_weight[iw:(iw + mul)]
#                 bias = self.affine_bias[ib:(ib + mul)]
#                 iw += mul
#                 ib += mul
#                 field = F.layer_norm(field, tuple((mul,)), weight, bias, self.eps)
#                 fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]
#                 continue
#
#             # For non-scalar features, use RMS value for std
#             field = field.reshape(-1, mul, d)  # [batch * sample, mul, repr]
#
#             if self.normalization == 'norm':
#                 field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
#             elif self.normalization == 'component':
#                 field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
#             else:
#                 raise ValueError("Invalid normalization option {}".format(self.normalization))
#             field_norm = torch.mean(field_norm, dim=1, keepdim=True)
#             field_norm = 1.0 / ((field_norm + self.eps).sqrt())  # [batch * sample, mul]
#
#             if self.affine:
#                 weight = self.affine_weight[None, iw:(iw + mul)]  # [1, mul]
#                 iw += mul
#                 field_norm = field_norm * weight  # [batch * sample, mul]
#             field = field * field_norm.reshape(-1, mul, 1)  # [batch * sample, mul, repr]
#
#             fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]
#
#         assert ix == dim
#
#         output = torch.cat(fields, dim=-1)
#         return output


def defect_site_in_batch(defect_site, natoms):
    cum_natoms = torch.cumsum(natoms, dim=0)
    cum_natoms = torch.cat([torch.tensor([0], device=natoms.device), cum_natoms])
    if type(defect_site) == list:
        cum_natoms = cum_natoms.tolist()
        defect_site_batch = []
        batch = []
        for i in range(len(defect_site)):
            for k in defect_site[i]:
                defect_site_batch.append(k + cum_natoms[i])
            batch.append(torch.zeros(len(defect_site[i])) + i)
        defect_site_batch = torch.tensor(defect_site_batch, device=natoms.device, dtype=torch.long)
        batch = torch.cat(batch, dim=-1)
        batch = batch.to(natoms.device).long()
    else:
        defect_site_batch = defect_site + cum_natoms[:-1]
        if defect_site_batch.max() >= torch.sum(natoms):
            print(defect_site_batch)
        batch = torch.arange(len(natoms), device=natoms.device)
    return defect_site_batch, batch


def create_large_graph(src_list, dst_list, defect_site_list, natoms):
    cum_natoms = torch.cumsum(natoms, dim=0)
    cum_natoms = torch.cat([torch.tensor([0], device=natoms.device), cum_natoms])[:-1].tolist()
    src_list = list(map(lambda x, y: torch.tensor(x) + y, src_list, cum_natoms))
    dst_list = list(map(lambda x, y: torch.tensor(x) + y, dst_list, cum_natoms))

    src = torch.cat(src_list, dim=-1)
    dst = torch.cat(dst_list, dim=-1)
    num_defect_sites = torch.tensor([len(d) for d in defect_site_list], device=natoms.device, dtype=torch.long)
    defect_site = list(map(lambda x, y: torch.tensor(x) + y, defect_site_list, cum_natoms))
    defect_batch = torch.repeat_interleave(torch.arange(len(natoms), device=natoms.device), num_defect_sites)
    defect_site = torch.cat(defect_site,dim=-1)
    return defect_batch, defect_site.to(natoms.device), src.to(natoms.device), dst.to(natoms.device)


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
    b = torch.linalg.cross(pos_ji, pos_kj).norm(dim=-1)
    angle = torch.atan2(b, a)
    return angle, idx_kj, idx_ji
