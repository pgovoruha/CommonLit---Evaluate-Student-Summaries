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


def create_criterion(criterion_config):

    if criterion_config.name == 'MCRMSELoss':
        return MCRMSELoss()
    elif criterion_config.name == 'RMSELoss':
        return RMSELoss()
    elif criterion_config.name == 'SmoothL1Loss':
        return nn.SmoothL1Loss(reduction='mean')

    return nn.MSELoss(reduction='mean')
