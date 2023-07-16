import pytorch_lightning as pl
import torch
import numpy as np
from commonlit.metrics.metric import mcrmse
from hydra.utils import instantiate
from typing import List, Dict
from pytorch_lightning.utilities.types import STEP_OUTPUT


def combine_values(key: str, outputs: List[Dict]) -> np.ndarray:
    return torch.cat([item[key] for item in outputs]).detach().cpu().numpy()


class LitModel(pl.LightningModule):

    def __init__(self, transformer_model, criterion, cfg_optimizer, cfg_scheduler,
                 learning_rate=2e-5):
        super().__init__()
        self.transformer_model = transformer_model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.cfg_optimizer = cfg_optimizer
        self.cfg_scheduler = cfg_scheduler
        self.save_hyperparameters()
        self.test_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask):
        return self.transformer_model(input_ids=input_ids, attention_mask=attention_mask)

    def _propagate_forward(self, batch) -> Dict:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['targets']
        predictions = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.criterion(predictions, targets)
        return {"loss": loss, "targets": targets, "predictions": predictions}

    def training_step(self, batch, batch_idx):

        outputs = self._propagate_forward(batch)
        loss = outputs['loss']
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        outputs = self._propagate_forward(batch)
        self.validation_step_outputs.append(outputs)

    def on_validation_epoch_end(self):
        loss = torch.stack([item['loss'] for item in self.validation_step_outputs])
        self.log('val_loss', loss.mean(), prog_bar=True)
        predictions = combine_values('predictions', self.validation_step_outputs)
        targets = combine_values('targets', self.validation_step_outputs)
        metric_mcrmse = mcrmse(targets, predictions)
        self.log('val_mcrmse', metric_mcrmse, prog_bar=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        outputs = self._propagate_forward(batch)
        self.test_step_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        loss = torch.stack([item['loss'] for item in self.test_step_outputs])
        self.log('test_loss', loss.mean(), prog_bar=True)
        predictions = combine_values('predictions', self.test_step_outputs)
        targets = combine_values('targets', self.test_step_outputs)
        metric_mcrmse = mcrmse(targets, predictions)
        self.log('test_mcrmse', metric_mcrmse)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg_optimizer, params=self.parameters(), lr=self.learning_rate)
        scheduler = instantiate(self.cfg_scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            'monitor': 'val_mcrmse',
            'interval': 'step'
        }

    def freeze_backbone(self):
        self.transformer_model.freeze_backbone()

    def unfreeze_backbone(self):
        self.transformer_model.unfreeze_backbone()
