import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, in_features, out_features, mid_features=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mid_features = mid_features
        self.linear1 = nn.Linear(in_features, mid_features, bias=True)
        self.linear2 = nn.Linear(mid_features, out_features, bias=True)
        self.activation1 = nn.Mish()
        self.activation2 = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        return x
