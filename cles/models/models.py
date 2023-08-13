import torch.nn as nn
from transformers import AutoModel, AutoConfig
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from cles.factories.factories import HeadFactory
from cles.factories.factories import PoolFactory
from cles.models.pools import MeanPooling, MaxPooling, MeanMaxPooling


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class CustomModel(BaseModel):

    def __init__(self,
                 cfg: DictConfig,
                 pool_factory: PoolFactory,
                 head_factory: HeadFactory):

        super().__init__()

        if cfg.config_path is not None:
            self.config = AutoConfig.from_pretrained(cfg.config_path)
        else:
            self.config = AutoConfig.from_pretrained(cfg.base_transformer)
        self.config.update({
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.0,
                "output_hidden_states": True
            })

        if cfg.config_path is not None:
            self.backbone = AutoModel.from_config(self.config)
        else:
            self.backbone = AutoModel.from_pretrained(cfg.base_transformer, config=self.config)

        self.sentence_transformer = AutoModel.from_pretrained(cfg.sentence_transformer)
        self.sentence_transformer.config.update({
            "output_hidden_states": True,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
        })
        self.pool = pool_factory.create_layer(self.config)
        self.sentence_transformer_pool = MaxPooling()
        self.head = head_factory.create_layer(backbone_config=self.config,
                                              sentence_transformer_config=self.sentence_transformer.config)
        # self.transformer = nn.Transformer(dropout=0, batch_first=True, d_model=self.config.hidden_size)
        for param in self.sentence_transformer.parameters():
            param.requires_grad = False

        if cfg.reinit_head_layer:
            for child in self.head.children():
                if isinstance(child, nn.Linear):
                    self._init_weights(child)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(3.0)

    def forward(self, inputs_prompt, inputs):
        outputs = self.backbone(**inputs)
        outputs_prompt = self.sentence_transformer(**inputs_prompt)

        outputs = self.pool(outputs, inputs)
        outputs_prompt = self.sentence_transformer_pool(outputs_prompt, inputs_prompt)

        # y = torch.abs(outputs - outputs_prompt)

        y = torch.cat([outputs_prompt, outputs], dim=1)

        y = self.head(y)
        return y

    def freeze_embeddings(self):
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

    def unfreeze_embeddings(self):
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = True

