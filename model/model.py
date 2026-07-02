import torch
from torch_geometric.nn.resolver import activation_resolver
from torch.nn import Module, ModuleList
from typing import Callable, Union,List
from common.utils import defect_site_in_batch, angles, create_large_graph
from common.graph_construction import graph_construction
from .dpp import InteractionPPBlock, EmbeddingBlock, OutputPPBlock, BesselBasisLayer, SphericalBasisLayer
from .readout import formation_energy_readout
from torch_scatter import scatter
from .muti_defect import defect_interaction


class defect_dpp(Module):
    def __init__(self,
                 hidden_channels: int,
                 out_emb_channels: int,
                 out_channels: int,
                 num_blocks: int,
                 int_emb_size: int,
                 basis_emb_size: int,
                 num_spherical: int,
                 num_radial: int,
                 cutoff: float = 5.0,
                 envelope_exponent: int = 5,
                 num_before_skip: int = 1,
                 num_after_skip: int = 1,
                 num_output_layers: int = 3,
                 act: Union[str, Callable] = 'swish',
                 compute_energy_per_sites=False,
                 contribution:List[str]=["V","G"],
                 ):
        super().__init__()
        self.contribution=contribution
        self.cutoff = cutoff
        self.compute_energy_per_sites = compute_energy_per_sites
        self.act = activation_resolver(act)
        self.rbf = BesselBasisLayer(num_radial, cutoff=self.cutoff,
                                    envelope_exponent=envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial,
                                       cutoff=self.cutoff,
                                       envelope_exponent=envelope_exponent)
        self.emb = EmbeddingBlock(num_radial, hidden_channels, self.act)
        self.interaction_blocks = ModuleList([
            InteractionPPBlock(
                hidden_channels,
                int_emb_size,
                basis_emb_size,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                self.act,
            ) for _ in range(num_blocks)
        ])
        self.output_blocks = torch.nn.ModuleList([
            OutputPPBlock(
                num_radial,
                hidden_channels,
                out_emb_channels,
                out_channels,
                num_output_layers,
                self.act,
            ) for _ in range(num_blocks + 1)
        ])
        self.readout = formation_energy_readout(out_channels, self.act)
        self.derivation1 = torch.nn.Linear(out_channels, out_channels, bias=False)
        self.derivation2 = torch.nn.Linear(out_channels, 1, bias=False)


    def _forward(self, z, j, i, dist, angle, idx_kj, idx_ji, defect_sites,output_defect=True):
        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        x = self.emb(z, rbf, i, j)
        if output_defect:
            P, D = self.output_blocks[0](x, rbf, i, defect_sites, output_defect)
            for interaction_block, output_block in zip(self.interaction_blocks, self.output_blocks[1:]):
                x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
                p, d = output_block(x, rbf, i, defect_sites,output_defect)
                P = P + p
                D = D + d
            return P, D
        else:
            P= self.output_blocks[0](x, rbf, i, defect_sites, output_defect)
            for interaction_block, output_block in zip(self.interaction_blocks, self.output_blocks[1:]):
                x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
                p= output_block(x, rbf, i, defect_sites,output_defect)
                P = P + p

            return P


    def forward(self, data):
        dist, vec, offsets, cell_offsets = data.dist, data.vec, data.offsets, data.cell_offsets
        defect_batch, defect_sites, j, i = create_large_graph(data.src, data.dst, defect_site_list=data.defect_sites,
                                                              natoms=data.natoms)
        if defect_sites.dim() == 2:
            defect_sites = defect_sites.squeeze(0)

        z, pos, cell, natoms = data.z, data.pos, data.cell, data.natoms
        z1 = z.clone()

        edge_index = torch.vstack([j, i])
        angle, idx_kj, idx_ji = angles(data.pos, edge_index, cell_offsets, num_nodes=len(data.z), use_pbc=True,
                                       offsets=offsets)
        z1[defect_sites] = data.removed_charges

        P_defect, defect_feat = self._forward(z, j, i, dist, angle, idx_kj, idx_ji, defect_sites,output_defect=True)


        if 0 in data.removed_charges:
            defect_batch_p, defect_sites_p,pristine_j,pristine_i =create_large_graph(data.pristine_src, data.pristine_dst, defect_site_list=data.defect_sites,natoms=data.natoms-1)
            pristine_edge_index = torch.vstack([pristine_j,pristine_i])
            pristine_angle, pristine_idx_kj, pristine_idx_ji = angles(data.pristine_pos, pristine_edge_index, data.pristine_cell_offset, num_nodes=len(data.pristine_z), use_pbc=True,
                                           offsets=data.pristine_offset)

            P_prisine = self._forward(data.pristine_z, pristine_j, pristine_i, data.pristine_dist, pristine_angle, pristine_idx_kj, pristine_idx_ji, defect_sites_p,output_defect=False)
            pristine_node_feats = torch.zeros_like(P_defect)
            node_index = torch.ones(len(z), dtype=torch.bool, device=z.device)
            node_index[defect_sites] = False
            pristine_node_feats[node_index] = P_prisine
            derivation = P_defect - pristine_node_feats
            defect_feat_out = self.readout(defect_node_feat=defect_feat, defect_site_batch=defect_batch)


        else:
            P_prisine, pristine_feat = self._forward(z1, j, i, dist, angle, idx_kj, idx_ji, defect_sites)
            derivation = P_defect - P_prisine
            defect_feat_out = self.readout(defect_node_feat=defect_feat-pristine_feat, defect_site_batch=defect_batch)
        neigh = self.derivation2(self.act(self.derivation1(derivation)))
        neigh_feat = scatter(neigh, dim=0, index=data.batch)

        if self.contribution==["V","G"] or self.contribution==["G","V"]:


            out = defect_feat_out+neigh_feat
        elif self.contribution==["G"]:
            out = neigh_feat
        elif self.contribution==["V"]:
            out=defect_feat_out
        else:
            raise ValueError
        if self.compute_energy_per_sites:
            out = out.squeeze(1) / data.num_defect_sites
        else:
            out = out.squeeze(1)

        return out
