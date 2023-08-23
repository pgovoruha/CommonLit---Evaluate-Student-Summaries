import torch.nn as nn
from transformers import AutoModel, AutoConfig
from omegaconf import DictConfig
import torch
from cles.factories.factories import HeadFactory
from cles.factories.factories import PoolFactory


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

        self.pool = pool_factory.create_layer(self.config)
        self.head = head_factory.create_layer(backbone_config=self.config)

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

    def forward(self, inputs):
        outputs = self.backbone(**inputs)
        outputs = self.pool(outputs, inputs)
        y = self.head(outputs)
        return y

    def freeze_embeddings(self):
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

    def unfreeze_embeddings(self):
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = True


class CustomModel2(CustomModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(in_features=self.cfg.embedding_size*2, out_features=self.config.hidden_size)

    def forward(self, text_embeddings, prompt_embeddings, inputs):
        outputs = self.backbone(**inputs)
        outputs = self.pool(outputs, inputs)
        outputs_embeddings = self.fc1(torch.cat([text_embeddings, prompt_embeddings], dim=1))
        y = torch.cat([outputs_embeddings, outputs], dim=1)
        y = self.head(y)
        return y


