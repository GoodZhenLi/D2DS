import torch
from torch.nn import Linear
from torch_scatter import scatter



class formation_energy_readout(torch.nn.Module):
    def __init__(self, out_channels, act):
        super().__init__()

        self.act = act
        self.lin_out1 = Linear(out_channels, 1)
        self.lin1 = Linear(out_channels, out_channels)
        self.lin2 = Linear(out_channels, out_channels)
    def forward(self, defect_node_feat, defect_site_batch):
        x = scatter(defect_node_feat, dim=0, index=defect_site_batch, reduce="sum")
        x = self.act(self.lin2(self.act(self.lin1(x)))) + x
        out = self.lin_out1(x)
        return out


class band_readout(torch.nn.Module):
    def __init__(self, out_channels, act):
        super(band_readout, self).__init__()
        self.act = act
        self.lin1 = Linear(out_channels, out_channels)
        self.lin2 = Linear(out_channels, out_channels)
        self.lin3 = Linear(out_channels, out_channels, bias=False)
        self.lin_conduct = Linear(out_channels, 1)
        self.lin_valence = Linear(out_channels, 1)

    def forward(self, defect_node_feat, defect_site_batch):
        defect_node_feat = torch.cat([defect_node_feat, ])
        x = self.act(self.lin2(self.act(self.lin1(defect_node_feat)))) + defect_node_feat
        x = self.lin3(x)
        conduct = abs(self.lin_conduct(x))
        valence = -abs(self.lin_valence(x))
        c_min = scatter(conduct, dim=0, index=defect_site_batch, reduce="min")
        v_max = scatter(valence, dim=0, index=defect_site_batch, reduce="max")
        out = c_min - v_max
        return out
