import torch
import torch.nn as nn


class Head(nn.Module):

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class OneLayerHead(Head):

    def __init__(self, in_features: int, out_features: int):

        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.linear(x)
        return x


class TwoLayerHead(Head):

    def __init__(self, in_features: int, out_features: int, middle_features: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=middle_features)
        self.linear_2 = nn.Linear(in_features=middle_features, out_features=out_features)
        self.relu = nn.ReLU()

    def forward(self, x):

        x_1 = self.linear_1(x)
        x_1 = self.relu(x_1)

        x = x + x_1
        x = self.linear_2(x)

        return x
