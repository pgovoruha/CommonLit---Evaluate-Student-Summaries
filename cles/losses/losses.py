import torch.nn as nn
import torch


class MCRMSELoss(nn.Module):
    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, y_true, y_pred):
        colwise_mse = torch.mean(torch.square(y_true - y_pred), dim=0)
        return torch.mean(torch.sqrt(colwise_mse), dim=0)


class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


class TwoLosses(nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.MSELoss()

    def forward(self, y_pred, y_true):

        loss1 = self.criterion1(y_pred[:, 0], y_true[:, 0])
        loss2 = self.criterion2(y_pred[:, 1], y_true[:, 1])

        return 0.3*loss1 + 0.7*loss2
