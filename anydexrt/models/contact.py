"""Contact model.
"""
import torch
from torch import nn

class ContactClassifier(nn.Module):
    def __init__(self, num_fingers=5, input_dim=6, hidden_dim=128):
        super(ContactClassifier, self).__init__()
        self.num_fingers = num_fingers
        self.nets = []
        for i in range(num_fingers - 1):
            net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.nets.append(net)
        self.nets = nn.ModuleList(self.nets)
    
    def forward(self, x):
        """x: [B, num_fingers-1, input_dim]"""
        outs = []
        for i in range(self.num_fingers - 1):
            out = self.nets[i](x[:, i])
            outs.append(out)
        outs = torch.cat(outs, dim=1)
        return outs
