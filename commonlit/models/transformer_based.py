import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from commonlit.models.pools import Pooling


class BaseModel(nn.Module):

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class TransformerWithCustomHead(BaseModel):

    def __init__(self, base_transformer: str, head: torch.nn.Module,
                 pool: Pooling,
                 config_path: str = None):

        super().__init__()
        self.config = AutoConfig.from_pretrained(config_path if config_path is not None else base_transformer)
        self.config.update({
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
        })
        self.base_transformer = AutoModel.from_pretrained(base_transformer, config=self.config) if config_path is None \
            else AutoModel.from_config(self.config)
        self.head = head
        self.pool = pool
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

    def forward(self, input_ids, attention_mask):
        outputs = self.base_transformer(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.pool(outputs[0], attention_mask)
        y = self.head(outputs)
        return y

    def freeze_backbone(self):
        for param in self.base_transformer.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.base_transformer.parameters():
            param.requires_grad = True


