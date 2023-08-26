from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from cles.models.pools import (Pooling, MeanPooling,
                               MaxPooling, MinPooling,
                               ConcatenatePooling, Conv1DPool,
                               CLSPooling, WeightedLayerPooling,
                               LSTMPooler, AttentionPooling, GeMPooling,
                               SimpleAttentionPooling, WKPooling, MeanMaxPooling, TwoAttentionPools)
from cles.models.heads import (OneLayerHead, TwoLayerHead,
                               Head, TwoSeparateHead, CNNHead,
                               OneLayerWithDropout, TwoLayerWithDropoutHead)
from cles.models.conc_layers import ConcatModel, AddModel, AttentionModel
from cles.losses.losses import (RMSELoss, TwoLosses, MCRMSELoss)
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.optim import Optimizer
from abc import ABC
import abc


class BaseNNFactory(ABC):

    @abc.abstractmethod
    def create_layer(self, *args, **kwargs) -> nn.Module:
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
        elif self.cfg.pool.name == 'Conv1DPool':
            return Conv1DPool(backbone_config=backbone_config, middle_dim=self.cfg.pool.params.middle_dim)
        elif self.cfg.pool.name == 'CLSPooling':
            return CLSPooling(layer_index=self.cfg.pool.params.layer_index)
        elif self.cfg.pool.name == 'WeightedLayerPooling':
            return WeightedLayerPooling(num_hidden_layers=backbone_config.num_hidden_layers,
                                        layer_start=self.cfg.pool.params.layer_start)
        elif self.cfg.pool.name == 'LSTMPooler':
            return LSTMPooler(num_layers=backbone_config.num_hidden_layers,
                              hidden_size=backbone_config.hidden_size,
                              hidden_dim_lstm=self.cfg.pool.params.hidden_dim_lstm)
        elif self.cfg.pool.name == 'AttentionPooling':
            return AttentionPooling(in_features=backbone_config.hidden_size,
                                    hidden_dim=backbone_config.hidden_size)
        elif self.cfg.pool.name == "TwoAttentionPools":
            return TwoAttentionPools(
                in_features=backbone_config.hidden_size,
                hidden_dim=backbone_config.hidden_size
            )
        elif self.cfg.pool.name == 'GeMPooling':
            return GeMPooling()
        elif self.cfg.pool.name == 'SimpleAttentionPooling':
            return SimpleAttentionPooling(in_features=backbone_config.hidden_size,
                                          middle_features=self.cfg.pool.params.middle_features)
        elif self.cfg.pool.name == 'WKPooling':
            return WKPooling(
                layer_start=self.cfg.pool.params.layer_start,
                context_window_size=self.cfg.pool.params.context_window_size
            )
        elif self.cfg.pool.name == 'MeanMaxPooling':
            return MeanMaxPooling()


class HeadFactory(BaseNNFactory):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_size_m = cfg.head.hidden_size_m
        if self.cfg.pool.name == 'ConcatenatePooling':
            self.hidden_size_m = self.cfg.head.hidden_size_m * self.cfg.pool.params.n_last_layers
        elif self.cfg.pool.name == "MeanMaxPooling":
            self.hidden_size_m = self.hidden_size_m * 2

    def create_layer(self, backbone_config: DictConfig) -> Head:

        if self.cfg.pool.name == 'LSTMPooler':
            in_features = self.cfg.pool.params.hidden_dim_lstm
        else:
            in_features = self.hidden_size_m * backbone_config.hidden_size + self.cfg.feature_vector_size

        if self.cfg.head.name == 'OneLayerHead':
            return OneLayerHead(in_features=in_features,
                                out_features=self.cfg.num_targets)
        elif self.cfg.head.name == 'OneLayerWithDropout':
            return OneLayerWithDropout(in_features=in_features,
                                       out_features=self.cfg.num_targets,
                                       dropout_p=self.cfg.head.params.dropout_p)
        elif self.cfg.head.name == 'TwoLayerHead':
            return TwoLayerHead(
                in_features=in_features,
                out_features=self.cfg.num_targets
            )
        elif self.cfg.head.name == 'TwoLayerWithDropoutHead':
            return TwoLayerWithDropoutHead(
                in_features=in_features,
                out_features=self.cfg.num_targets,
                dropout_1=self.cfg.head.params.dropout_1,
                dropout_2=self.cfg.head.params.dropout_2
            )
        elif self.cfg.head.name == 'TwoSeparateHead':
            return TwoSeparateHead(
                in_features=self.cfg.head.hidden_size_m * backbone_config.hidden_size,
                out_features=1
            )
        elif self.cfg.head.name == 'CNNHead':

            if self.cfg.pool.name != 'Conv1DPool':
                raise ValueError(f'CNNHead only works with Cov1DPool, received : {self.cfg.pool.name}')

            return CNNHead(
                in_features=self.cfg.pool.params.middle_dim,
                out_features=self.cfg.num_targets
            )


class CriterionFactory(BaseNNFactory):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def create_layer(self, backbone_config: DictConfig = None) -> nn.Module:

        if self.cfg.criterion.name == 'RMSELoss':
            return RMSELoss()
        elif self.cfg.criterion.name == 'TwoLosses':
            return TwoLosses()
        elif self.cfg.criterion.name == "MCRMSELoss":
            return MCRMSELoss()


class OptimizerFactory:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def return_optimizer(self, params, lr) -> Optimizer:

        if self.cfg.optim.name == 'Adam':
            if self.cfg.optim.params is not None:
                return torch.optim.Adam(params=params, lr=lr, **self.cfg.optim.params)
            else:
                return torch.optim.Adam(params=params, lr=lr)
        elif self.cfg.optim.name == 'AdamW':
            if self.cfg.optim.params is not None:
                return torch.optim.AdamW(params=params, lr=lr, **self.cfg.optim.params)
            else:
                return torch.optim.AdamW(params=params, lr=lr)


class SchedulerFactory:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def return_scheduler(self, optimizer, num_train_steps=None):

        if self.cfg.scheduler.name == 'ReduceLROnPlateau':
            if self.cfg.scheduler.params is not None:
                return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, **self.cfg.scheduler.params)
            else:
                return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
        elif self.cfg.scheduler.name == 'Cosine':
            if self.cfg.scheduler.params is not None:
                return get_cosine_schedule_with_warmup(
                    optimizer=optimizer, num_training_steps=num_train_steps, **self.cfg.scheduler.params
                )
            else:
                return get_cosine_schedule_with_warmup(
                    optimizer=optimizer, num_training_steps=num_train_steps
                )
        elif self.cfg.scheduler.name == "Linear":
            if self.cfg.scheduler.params is not None:
                return get_linear_schedule_with_warmup(optimizer=optimizer,
                                                       num_training_steps=num_train_steps,
                                                       **self.cfg.scheduler.params)
            else:
                return get_linear_schedule_with_warmup(optimizer=optimizer,
                                                       num_training_steps=num_train_steps)


class CombineFeaturesFactory(BaseNNFactory):
    def __init__(self, cfg):
        self.cfg = cfg

    def create_layer(self, backbone_config):

        if self.cfg.combine_layer_name == "ConcatModel":
            return ConcatModel(hidden_size=backbone_config.hidden_size, feature_size=self.cfg.feature_vector_size)
        elif self.cfg.combine_layer_name == "AddModel":
            return AddModel(hidden_size=backbone_config.hidden_size, feature_size=self.cfg.feature_vector_size)
        elif self.cfg.combine_layer_name == 'AttentionModel':
            return AttentionModel(hidden_size=backbone_config.hidden_size, feature_size=self.cfg.feature_vector_size)


