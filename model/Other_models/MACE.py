###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from torch.nn import SiLU
from mace.modules.embeddings import GenericJointEmbedding
from mace.modules.radial import ZBLBasis
from mace.tools.scatter import scatter_mean, scatter_sum
from mace.tools.torch_tools import get_change_of_basis, spherical_to_cartesian

from mace.modules.blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipolePolarReadoutBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipolePolarReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from mace.modules.blocks import RealAgnosticResidualNonLinearInteractionBlock
from mace.modules.utils import (
    compute_dielectric_gradients,
    compute_fixed_charge_dipole,
    compute_fixed_charge_dipole_polar,
    get_atomic_virials_stresses,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
    prepare_graph,
)

from common.utils import create_large_graph

import torch.nn.functional as F
def atomic_to_one_hot(atomic_numbers, max_atomic_number=94, device=None, dtype=torch.float32):
    
    # 转换为 torch 张量（long 类型），并推断或指定设备
    if not isinstance(atomic_numbers, torch.Tensor):
        atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.long)

    if device is None:
        device = atomic_numbers.device
    else:
        device = torch.device(device)
        atomic_numbers = atomic_numbers.to(device)



    one_hot = F.one_hot(atomic_numbers, num_classes=max_atomic_number).to(dtype)
    return one_hot




pbe_energy={"0":0,
        "1": -1.11734008,
        "2": 0.00096759,
        "3": -0.29754725,
        "4": -0.01781697,
        "5": -0.26885011,
        "6": -1.26173507,
        "7": -3.12438806,
        "8": -1.54838784,
        "9": -0.51882044,
        "10": -0.01241601,
        "11": -0.22883163,
        "12": -0.00951015,
        "13": -0.21630193,
        "14": -0.8263903,
        "15": -1.88816619,
        "16": -0.89160769,
        "17": -0.25828273,
        "18": -0.04925973,
        "19": -0.22697913,
        "20": -0.0927795,
        "21": -2.11396364,
        "22": -2.50054871,
        "23": -3.70477179,
        "24": -5.60261985,
        "25": -5.32541181,
        "26": -3.52004933,
        "27": -1.93555024,
        "28": -0.9351969,
        "29": -0.60025846,
        "30": -0.1651332,
        "31": -0.32990651,
        "32": -0.77971828,
        "33": -1.68367812,
        "34": -0.76941032,
        "35": -0.22213843,
        "36": -0.0335879,
        "37": -0.1881724,
        "38": -0.06826294,
        "39": -2.17084228,
        "40": -2.28579303,
        "41": -3.13429753,
        "42": -4.60211419,
        "43": -3.45201492,
        "44": -2.38073513,
        "45": -1.46855515,
        "46": -1.4773126,
        "47": -0.33954585,
        "48": -0.16843877,
        "49": -0.35470981,
        "50": -0.83642657,
        "51": -1.41101987,
        "52": -0.65740879,
        "53": -0.18964571,
        "54": -0.00857582,
        "55": -0.13771876,
        "56": -0.03457659,
        "57": -0.45580806,
        "58": -1.3309175,
        "59": -0.29671824,
        "60": -0.30391193,
        "61": -0.30898427,
        "62": -0.25470891,
        "63": -8.38001538,
        "64": -10.38896525,
        "65": -0.3059505,
        "66": -0.30676216,
        "67": -0.30874667,
        "69": -0.25190039,
        "70": -0.06431414,
        "71": -0.31997586,
        "72": -3.52770927,
        "73": -3.54492209,
        "75": -4.70108713,
        "76": -2.88257209,
        "77": -1.46779304,
        "78": -0.50269936,
        "79": -0.28801193,
        "80": -0.12454674,
        "81": -0.31737194,
        "82": -0.77644932,
        "83": -1.32627283,
        "89": -0.26827152,
        "90": -0.90817426,
        "91": -2.47653193,
        "92": -4.90438537,
        "93": -7.63378961,
        "94": -10.77237713
    }
atomic_energies=[]

for i in range(95):
    if str(i) in pbe_energy.keys():
        atomic_energies.append(pbe_energy[str(i)])
    else:
        atomic_energies.append(0)

atomic_energies=np.asarray(atomic_energies)
atomic_numbers=np.arange(len(atomic_energies))
@compile_mode("script")
class MACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float=5.0,
        num_bessel: int=7,
        num_polynomial_cutoff: int=6,
        max_ell: int=2,
        interaction_cls: InteractionBlock=RealAgnosticResidualNonLinearInteractionBlock,
        interaction_cls_first: InteractionBlock=RealAgnosticResidualNonLinearInteractionBlock,
        num_interactions:int= 4,
        num_elements: int=len(atomic_energies),
        hidden_irreps: o3.Irreps=o3.Irreps([(128, (0, 1)), (128, (1, -1)),(128, (2, 1))]),
        MLP_irreps: o3.Irreps=o3.Irreps([(256, (0, 1))]),
        atomic_energies: np.ndarray=atomic_energies,
        avg_num_neighbors: float=19,
        atomic_numbers: List[int]=atomic_numbers,
        correlation: Union[int, List[int]]=2,
        gate: Optional[Callable]=torch.nn.functional.silu,
        pair_repulsion: bool = False,
        apply_cutoff: bool = True,
        use_reduced_cg: bool = True,
        use_so3: bool = False,
        use_agnostic_product: bool = False,
        use_last_readout_only: bool = False,
        use_embedding_readout: bool = False,
        distance_transform: str = "None",
        edge_irreps: Optional[o3.Irreps] = None,
        use_edge_irreps_first: bool = False,
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,
        embedding_specs: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
        lammps_mliap: Optional[bool] = False,
        readout_cls: Optional[Type[NonLinearReadoutBlock]] = NonLinearReadoutBlock,
        keep_last_layer_irreps: bool = False,
            compute_energy_per_sites: bool = False,
            compute_defect_only: bool =True,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if heads is None:
            heads = ["Default"]
        self.heads = heads
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        self.lammps_mliap = lammps_mliap
        self.apply_cutoff = apply_cutoff
        self.edge_irreps = edge_irreps
        self.use_reduced_cg = use_reduced_cg
        self.use_agnostic_product = use_agnostic_product
        self.use_so3 = use_so3
        self.use_last_readout_only = use_last_readout_only
        self.use_edge_irreps_first = use_edge_irreps_first
        self.compute_energy_per_sites=compute_energy_per_sites
        self.compute_defect_only=compute_defect_only
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        )
        self.num_elements=num_elements
        embedding_size = node_feats_irreps.count(o3.Irrep(0, 1))
        if embedding_specs is not None:
            self.embedding_specs = embedding_specs
            self.joint_embedding = GenericJointEmbedding(
                base_dim=embedding_size,
                embedding_specs=embedding_specs,
                out_dim=embedding_size,
            )
            if use_embedding_readout:
                self.embedding_readout = LinearReadoutBlock(
                    node_feats_irreps,
                    o3.Irreps(f"{len(heads)}x0e"),
                    cueq_config,
                    oeq_config,
                )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
            apply_cutoff=apply_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(p=num_polynomial_cutoff)
            self.pair_repulsion = True

        if not use_so3:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p=1)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))

        # interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        def generate_irreps(l):
            str_irrep = "+".join([f"1x{i}e+1x{i}o" for i in range(l + 1)])
            return o3.Irreps(str_irrep)

        sh_irreps_inter = sh_irreps
        if hidden_irreps.count(o3.Irrep(0, -1)) > 0:
            sh_irreps_inter = generate_irreps(max_ell)
        interaction_irreps = (sh_irreps_inter * num_features).sort()[0].simplify()
        interaction_irreps_first = (sh_irreps * num_features).sort()[0].simplify()

        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)
        if num_interactions == 1:
            hidden_irreps_out = str(hidden_irreps[0])
        else:
            hidden_irreps_out = hidden_irreps
        edge_irreps_first = None
        if use_edge_irreps_first and edge_irreps is not None:
            edge_irreps_first = o3.Irreps(f"{edge_irreps.count(o3.Irrep(0, 1))}x0e")
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps_first,
            hidden_irreps=hidden_irreps_out,
            edge_irreps=edge_irreps_first,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps_out,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
            use_reduced_cg=use_reduced_cg,
            use_agnostic_product=use_agnostic_product,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        if not use_last_readout_only:
            self.readouts.append(
                LinearReadoutBlock(
                    hidden_irreps_out,
                    o3.Irreps(f"{len(heads)}x0e"),
                    cueq_config,
                    oeq_config,
                )
            )

        for i in range(num_interactions - 1):
            if i == num_interactions - 2 and not keep_last_layer_irreps:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                edge_irreps=edge_irreps,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
                use_reduced_cg=use_reduced_cg,
                use_agnostic_product=use_agnostic_product,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    readout_cls(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{len(heads)}x0e"),
                        len(heads),
                        cueq_config,
                        oeq_config,
                    )
                )
            elif not use_last_readout_only:
                self.readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps,
                        o3.Irreps(f"{len(heads)}x0e"),
                        cueq_config,
                        oeq_config,
                    )
                )

    def forward(
        self,
        data,

    ):
        # Setup
        # ctx = prepare_graph(
        #     data,
        #     compute_virials=compute_virials,
        #     compute_stress=compute_stress,
        #     compute_displacement=compute_displacement,
        #     lammps_mliap=lammps_mliap,
        # )
        is_lammps = False
        lammps_class=None
        # num_atoms_arange = ctx.num_atoms_arange.to(torch.int64)
        # num_graphs = ctx.num_graphs
        # displacement = ctx.displacement
        # positions = ctx.positions
        # vectors = ctx.vectors
        # lengths = ctx.lengths
        # cell = ctx.cell
        # node_heads = ctx.node_heads.to(torch.int64)
        # interaction_kwargs = ctx.interaction_kwargs
        # lammps_natoms = interaction_kwargs.lammps_natoms
        # lammps_class = interaction_kwargs.lammps_class

        node_heads = torch.zeros_like(data.batch)
        lammps_natoms = (0, 0)
        num_atoms_arange=torch.arange(len(data.z),device=data.batch.device)

        lengths, vectors, offsets, cell_offsets = data.dist.view(-1,1), data.vec, data.offsets, data.cell_offsets
        defect_batch, defect_sites, src, dst = create_large_graph(data.src, data.dst, defect_site_list=data.defect_sites,
                                                              natoms=data.natoms)
        if defect_sites.dim() == 2:
            defect_sites = defect_sites.squeeze(0)

        z, pos, cell, natoms = data.z, data.pos, data.cell, data.natoms


        edge_index = torch.vstack([src, dst])

        num_graphs=len(natoms)
        node_attrs=atomic_to_one_hot(z,max_atomic_number=self.num_elements)
        # Atomic energies
        node_e0 = self.atomic_energies_fn(node_attrs)[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data.batch, dim=0, dim_size=num_graphs
        ).to(
            vectors.dtype
        )  # [n_graphs, n_heads]







        # Embeddings
        node_feats = self.node_embedding(node_attrs)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, node_attrs, edge_index, self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, node_attrs, edge_index, self.atomic_numbers
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
            pair_energy = scatter_sum(
                src=pair_node_energy, index=data.batch, dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
        else:
            pair_node_energy = torch.zeros_like(node_e0)
            pair_energy = torch.zeros_like(e0)

        if hasattr(self, "joint_embedding"):
            embedding_features: Dict[str, torch.Tensor] = {}
            for name, _ in self.embedding_specs.items():
                embedding_features[name] = data[name]
            node_feats += self.joint_embedding(
                data.batch,
                embedding_features,
            )
            if hasattr(self, "embedding_readout"):
                embedding_node_energy = self.embedding_readout(
                    node_feats, node_heads
                ).squeeze(-1)
                embedding_energy = scatter_sum(
                    src=embedding_node_energy,
                    index=data.batch,
                    dim=0,
                    dim_size=num_graphs,
                )
                e0 += embedding_energy

        # Interactions

        node_energies_list = [node_e0, pair_node_energy]
        node_feats_concat: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            node_attrs_slice = node_attrs
            if is_lammps and i > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                cutoff=cutoff,
                first_layer=(i == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and i == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_concat.append(node_feats)

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es = readout(node_feats_concat[feat_idx], node_heads)[num_atoms_arange,node_heads]
            if self.compute_defect_only:
                # energies = [e0, pair_energy]
                energies = []
                energy = scatter_sum(node_es[defect_sites], defect_batch, dim=0, dim_size=num_graphs)
            else:
                # energies = [e0, pair_energy]
                energies = []
                energy = scatter_sum(node_es, data.batch, dim=0, dim_size=num_graphs)
            energies.append(energy)
            node_energies_list.append(node_es)

        # if self.compute_defect_only:
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)
        # node_energy = torch.sum(torch.stack(node_energies_list, dim=-1), dim=-1)
        # node_feats_out = torch.cat(node_feats_concat, dim=-1)

        if self.compute_energy_per_sites:
            out = total_energy/ data.num_defect_sites
        else:
            out = total_energy

        return out


        # return {
        #     "energy": total_energy,
        #     "node_energy": node_energy,
        #     "contributions": contributions,
        #     "node_feats": node_feats_out,
        # }

