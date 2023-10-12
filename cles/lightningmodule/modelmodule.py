import lightning as L
import torch
import torch.nn as nn
import numpy as np
from cles.metrics.metric import mcrmse
from cles.losses.losses import create_criterion
from typing import List, Dict
from omegaconf import DictConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from cles.utils import collate, combine_values


def get_scheduler(optimizer, scheduler_config, num_training_steps):

    if scheduler_config.name == 'Cosine':
        return get_cosine_schedule_with_warmup(optimizer=optimizer,
                                               num_training_steps=num_training_steps,
                                               num_warmup_steps=int(scheduler_config.params.num_warmup_steps),
                                               num_cycles=float(scheduler_config.params.num_cycles))

    elif scheduler_config.name == 'Linear':
        return get_linear_schedule_with_warmup(optimizer=optimizer,
                                               num_training_steps=num_training_steps,
                                               num_warmup_steps=int(scheduler_config.params.num_warmup_steps))


class LitModel(L.LightningModule):

    def __init__(self, transformer_model: nn.Module,
                 cfg: DictConfig
                 ):
        super().__init__()
        self.transformer_model = transformer_model
        self.backbone_lr = float(cfg.train.backbone_lr)
        self.head_lr = float(cfg.train.head_lr)
        self.weight_decay = float(cfg.train.optimizer_config.weight_decay)
        # self.learning_rate = float(cfg.train.learning_rate)
        self.scheduler_config = cfg.train.scheduler
        self.optimizer_config = cfg.train.optimizer_config
        self.criterion = create_criterion(cfg.train.criterion)
        self.save_hyperparameters()
        self.test_step_outputs = []
        self.validation_step_outputs = []
        self.best_val_mcrmse = 1

    def forward(self, inputs):
        return self.transformer_model(inputs)

    def _propagate_forward(self, batch) -> Dict:
        inputs, targets = batch
        inputs = collate(inputs)
        predictions = self(inputs)
        loss = self.criterion(predictions, targets)
        return {"loss": loss, "targets": targets, "predictions": predictions}

    def _log_metrics(self, step_outputs: List, batch_type: str):

        loss = torch.stack([item['loss'] for item in step_outputs])

        self.log(f'{batch_type}_loss', loss.mean(), prog_bar=True)

        predictions = combine_values('predictions', step_outputs)
        targets = combine_values('targets', step_outputs)

        metric_mcrmse, scores = mcrmse(targets, predictions)

        self.log(f'{batch_type}_mcrmse', metric_mcrmse, prog_bar=True)
        if batch_type == 'val':
            if metric_mcrmse < self.best_val_mcrmse:
                self.best_val_mcrmse = metric_mcrmse
            self.log('best_val_mcrmse', self.best_val_mcrmse, prog_bar=True)

        if len(scores) > 1:
            self.log(f'{batch_type}_cont_rmse', scores[0], prog_bar=True)
            self.log(f'{batch_type}_word_rmse', scores[1], prog_bar=True)

    def training_step(self, batch, batch_idx):
        outputs = self._propagate_forward(batch)
        loss = outputs['loss']

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self._propagate_forward(batch)
        self.validation_step_outputs.append(outputs)

    def on_validation_epoch_end(self):
        self._log_metrics(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        outputs = self._propagate_forward(batch)
        self.test_step_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        self._log_metrics(self.test_step_outputs, 'test')
        self.test_step_outputs.clear()

    def get_optimizer_params(self):
        all_params = list(self.transformer_model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        params_list = []

        backbone_params_no_decay = [p for n, p in all_params if ('backbone' in n) and
                                    (any(nd in n for nd in no_decay))]
        params_list.append({
            'params': backbone_params_no_decay,
            'lr': self.backbone_lr,
            'weight_decay': 0.0
        })

        backbone_params_decay = [p for n, p in all_params if ('backbone' in n) and
                                 (not any(nd in n for nd in no_decay))]
        params_list.append({
            'params': backbone_params_decay,
            'lr': self.backbone_lr,
            'weight_decay': self.weight_decay
        })

        head_params_no_decay = [p for n, p in all_params if ('head' in n) and
                                (any(nd in n for nd in no_decay))]
        params_list.append({
            'params': head_params_no_decay,
            'lr': self.head_lr,
            'weight_decay': 0.0
        })

        head_params_decay = [p for n, p in all_params if ('head' in n) and
                             (not any(nd in n for nd in no_decay))]
        params_list.append({
            'params': head_params_decay,
            'lr': self.head_lr,
            'weight_decay': self.weight_decay
        })

        return params_list

    def configure_optimizers(self):

        params = self.get_optimizer_params()
        optimizer = torch.optim.AdamW(params=params)
        scheduler = get_scheduler(optimizer=optimizer, scheduler_config=self.scheduler_config,
                                  num_training_steps=int(self.trainer.estimated_stepping_batches))

        return [optimizer], [{"scheduler": scheduler,
                              "interval": "step"}]


class LitModelWithPromptText(LitModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, prompt_inputs, inputs):

        return self.transformer_model(prompt_inputs, inputs)

    def _propagate_forward(self, batch) -> Dict:
        prompt_inputs, inputs, targets = batch
        inputs = collate(inputs)
        prompt_inputs = collate(prompt_inputs)
        predictions = self(prompt_inputs, inputs)
        loss = self.criterion(predictions, targets)
        return {"loss": loss, "targets": targets, "predictions": predictions}


def get_lit_model(cfg, custom_model):

    if cfg.train.model_name == 'CustomModel':
        lit_model = LitModel(
            transformer_model=custom_model,
            cfg=cfg
        )
    elif cfg.train.model_name == 'CustomModelWithPromptText':
        lit_model = LitModelWithPromptText(
            transformer_model=custom_model,
            cfg=cfg
        )
    else:
        raise ValueError(f'Unspecified train model name {cfg.train.model_name}')

    return lit_model

