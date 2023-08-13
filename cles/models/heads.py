import torch
import torch.nn as nn


class Head(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class OneLayerHead(Head):

    def __init__(self, in_features: int, out_features: int):

        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        # self.norm = nn.LayerNorm(in_features)

    def forward(self, x):
        # x = self.norm(x)
        x = self.linear(x)
        return x


class OneLayerWithDropout(Head):

    def __init__(self, in_features: int, out_features: int, dropout_p=0.1):

        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x


class TwoSeparateHead(Head):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.head1 = OneLayerHead(in_features=in_features, out_features=out_features)
        self.head2 = OneLayerHead(in_features=in_features, out_features=out_features)

    def forward(self, x):
        if isinstance(x, tuple):
            y1 = self.head1(x[0])
            y2 = self.head2(x[1])
        else:
            y1 = self.head1(x)
            y2 = self.head2(x)

        return torch.cat([y1, y2], dim=1)


class TwoLayerHead(Head):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=in_features)
        self.linear_2 = nn.Linear(in_features=in_features, out_features=out_features)
        self.relu = nn.ReLU()

    def forward(self, x):

        x_1 = self.linear_1(x)
        x_1 = self.relu(x_1)

        x = x + x_1
        x = self.linear_2(x)

        return x


class TwoLayerWithDropoutHead(Head):

    def __init__(self, in_features: int, out_features: int, dropout_1=0.2, dropout_2=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=in_features)
        self.linear_2 = nn.Linear(in_features=in_features, out_features=out_features)
        self.dropout1 = nn.Dropout(dropout_1)
        self.dropout2 = nn.Dropout(dropout_2)
        self.relu = nn.ReLU()

    def forward(self, x):

        x_1 = self.dropout1(x)
        x_1 = self.linear_1(x_1)
        x_1 = self.relu(x_1)

        x = x + x_1
        x = self.dropout2(x)
        x = self.linear_2(x)

        return x


class CNNHead(Head):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.cnn = nn.Conv1d(in_features, out_features, kernel_size=2, padding=1)

    def forward(self, x):
        x = self.cnn(x)
        x, _ = torch.max(x, 2)
        return x
