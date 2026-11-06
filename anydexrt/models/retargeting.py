"""Retargeting model.
"""
import torch
from torch import nn

class MappingModel(nn.Module):
    def __init__(self, finger_num):
        super().__init__()
        self.nets = []
        self.final = []
        self.finger_num = finger_num
        for i in range( self.finger_num):
            net = nn.Sequential(
                nn.Linear(3, 128),
                nn.LeakyReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 128),
                nn.LeakyReLU(),
                nn.BatchNorm1d(128),
            )
            self.nets.append(net)
            final = nn.Sequential(
                nn.Linear(128, 3),
                nn.Tanh()
            )
            self.final.append(final)
        self.nets = nn.ModuleList(self.nets)
        self.final = nn.ModuleList(self.final)

    def forward(self, keypoints):
        dim = 3
        keypoints = keypoints.view(-1, self.finger_num, dim)

        feats = []
        outs = []
        for i in range( self.finger_num):
            feat = self.nets[i](keypoints[:, i])
            feats.append(feat)
            out = self.final[i](feat)
            outs.append(out)
        feats = torch.stack(feats, dim=1)
        outs = torch.stack(outs, dim=1)
        return outs, feats
