import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import AutoModel, AutoConfig
from omegaconf import DictConfig
from cles.models.pools import create_pooling


class CustomTransformerOutput:

    def __init__(self, last_hidden_state, hidden_states):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class CustomModel(BaseModel):

    def __init__(self,
                 cfg: DictConfig):

        super().__init__()
        self.cfg = cfg
        if cfg.backbone.config_path is not None:
            self.backbone_config = AutoConfig.from_pretrained(cfg.backbone.config_path)
        else:
            self.backbone_config = AutoConfig.from_pretrained(cfg.backbone.name)
        self.backbone_config.update({
            "hidden_dropout": cfg.backbone.hidden_dropout,
            "hidden_dropout_prob": cfg.backbone.hidden_dropout_prob,
            "attention_dropout": cfg.backbone.attention_dropout,
            "attention_probs_dropout_prob": cfg.backbone.attention_probs_dropout_prob,
            "output_hidden_states": True
        })

        if cfg.backbone.config_path is not None:
            self.backbone = AutoModel.from_config(self.backbone_config)
        else:
            self.backbone = AutoModel.from_pretrained(cfg.backbone.name, config=self.backbone_config)

        self.pool = create_pooling(self.backbone_config, self.cfg.pool)
        self.head = nn.Linear(in_features=self.pool.output_dim, out_features=cfg.num_targets)

        if cfg.train.reinit_head_layer:
            self._init_weights(self.head)

        if cfg.backbone.freeze_embeddings:
            self.freeze_embeddings()

        if cfg.backbone.freeze_n_layers is not None:
            self.freeze_n_layers(cfg.backbone.freeze_n_layers)

        if cfg.train.enable_gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.backbone_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.backbone_config.initializer_range)
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

    def freeze_n_layers(self, n):
        for k, param in self.backbone.encoder.layer.named_parameters():
            l_num = int(k.split(".")[0])
            if l_num < n:
                param.requires_grad = False


class CustomModelWithPromptText(CustomModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = self.cfg.train.window_size
        self.overlap_size = self.cfg.train.overlap_size
        self.max_len = self.cfg.train.max_len

    def combine(self, prompt_inputs, prompt_outputs, inputs, outputs):
        last_hidden_state = torch.cat([outputs.last_hidden_state,
                                       prompt_outputs.last_hidden_state], dim=1)
        hidden_states = [torch.cat((hs1, hs2), dim=1) for hs1, hs2 in zip(outputs.hidden_states,
                                                                          prompt_outputs.hidden_states)]
        attention_mask = torch.cat([inputs['attention_mask'], prompt_inputs['attention_mask']], dim=1)

        output = CustomTransformerOutput(last_hidden_state=last_hidden_state,
                                         hidden_states=hidden_states)
        inputs['attention_mask'] = attention_mask

        return output, inputs

    def forward(self, prompt_inputs, inputs):
        prompt_outputs = self.backbone(**prompt_inputs)
        outputs = self.backbone(**inputs)

        outputs, inputs = self.combine(prompt_inputs, prompt_outputs, inputs, outputs)

        outputs = self.pool(outputs, inputs)
        outputs = self.head(outputs)
        return outputs


def get_model(config: DictConfig):

    if config.train.model_name == 'CustomModel':

        return CustomModel(cfg=config)
    elif config.train.model_name == 'CustomModelWithPromptText':
        return CustomModelWithPromptText(cfg=config)
    else:
        raise ValueError(f'Unknown model_name : {config.train.model_name}')
