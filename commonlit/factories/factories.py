from commonlit.models.pools import (Pooling, MeanPooling,
                                    MaxPooling, MinPooling,
                                    ConcatenatePooling)
from commonlit.models.heads import OneLayerHead, TwoLayerHead, Head
from commonlit.losses.losses import RMSELoss
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.optim import Optimizer
from abc import ABC
import abc


class BaseNNFactory(ABC):

    @abc.abstractmethod
    def create_layer(self, backbone_config: DictConfig) -> nn.Module:
        pass


class PoolFactory(BaseNNFactory):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def create_layer(self, backbone_config: DictConfig) -> Pooling:

        if self.cfg.pool.name == 'MeanPooling':
            return MeanPooling()
        elif self.cfg.pool.name == 'MaxPooling':
            return MaxPooling()
        elif self.cfg.pool.name == 'MinPooling':
            return MinPooling()
        elif self.cfg.pool.name == 'ConcatenatePooling':
            return ConcatenatePooling(n_last_layers=self.cfg.pool.params.n_last_layers)


class HeadFactory(BaseNNFactory):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        if self.cfg.pool.name == 'ConcatenatePooling':
            self.cfg.head.hidden_size_m = self.cfg.head.hidden_size_m * self.cfg.pool.params.n_last_layers

    def create_layer(self, backbone_config: DictConfig) -> Head:

        if self.cfg.head.name == 'OneLayerHead':
            return OneLayerHead(in_features=self.cfg.head.hidden_size_m * backbone_config.hidden_size,
                                out_features=self.cfg.num_targets)
        elif self.cfg.head.name == 'TwoLayerHead':
            if self.cfg.head.middle_features is not None:
                middle_features = self.cfg.head.middle_features
            else:
                middle_features = backbone_config.hidden_size * self.cfg.head.hidden_size_m
            return TwoLayerHead(
                in_features=self.cfg.head.hidden_size_m * backbone_config.hidden_size,
                out_features=self.cfg.num_targets,
                middle_features=middle_features
            )


class CriterionFactory(BaseNNFactory):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def create_layer(self, backbone_config: DictConfig = None) -> nn.Module:

        if self.cfg.criterion.name == 'RMSELoss':
            return RMSELoss()


class OptimizerFactory:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def return_optimizer(self, *args, **kwargs) -> Optimizer:

        if self.cfg.optim.name == 'Adam':
            if self.cfg.optim.params is not None:
                return torch.optim.Adam(*args, **kwargs, **self.cfg.optim.params)
            else:
                return torch.optim.Adam(*args, **kwargs)


class SchedulerFactory:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def return_scheduler(self, *args, **kwargs):

        if self.cfg.scheduler.name == 'ReduceLROnPlateau':
            if self.cfg.scheduler.params is not None:
                return torch.optim.lr_scheduler.ReduceLROnPlateau(*args, **kwargs, **self.cfg.scheduler.params)
            else:
                return torch.optim.lr_scheduler.ReduceLROnPlateau(*args, **kwargs)



