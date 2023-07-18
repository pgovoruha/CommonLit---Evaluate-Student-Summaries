import torch
import torch.nn as nn


class Head(nn.Module):

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class CustomHead(Head):

    def __init__(self, in_features: int, out_features: int, dropout_1=0.2, dropout_2=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=in_features)
        self.linear_2 = nn.Linear(in_features=in_features, out_features=out_features)
        # self.dropout1 = nn.Dropout(dropout_1)
        # self.dropout2 = nn.Dropout(dropout_2)
        self.relu = nn.ReLU()

    def forward(self, x):

        # x_1 = self.dropout1(x)
        x_1 = self.linear_1(x)
        x_1 = self.relu(x_1)

        x = x + x_1
        # x = self.dropout2(x)
        x = self.linear_2(x)

        return x
