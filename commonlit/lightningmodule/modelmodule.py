import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from commonlit.metrics.metric import mcrmse
from commonlit.factories.factories import OptimizerFactory, SchedulerFactory, CriterionFactory
from hydra.utils import instantiate
from typing import List, Dict
from pytorch_lightning.utilities.types import STEP_OUTPUT


def combine_values(key: str, outputs: List[Dict]) -> np.ndarray:
    return torch.cat([item[key] for item in outputs]).detach().cpu().numpy()


class LitModel(pl.LightningModule):

    def __init__(self, transformer_model: nn.Module,
                 scheduler_factory: SchedulerFactory,
                 optimizer_factory: OptimizerFactory,
                 criterion_factory: CriterionFactory,
                 learning_rate: float = 2e-5,
                 frequency: int = 50):
        super().__init__()
        self.transformer_model = transformer_model
        self.criterion = criterion_factory.create_layer()
        self.learning_rate = learning_rate
        self.scheduler_factory = scheduler_factory
        self.optimizer_factory = optimizer_factory
        self.frequency = frequency
        self.save_hyperparameters()
        self.test_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, inputs):
        return self.transformer_model(inputs)

    def _propagate_forward(self, batch) -> Dict:
        inputs, targets = batch
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

    def configure_optimizers(self):

        optimizer = self.optimizer_factory.return_optimizer(params=self.transformer_model.parameters(),
                                                            lr=self.learning_rate)

        scheduler = self.scheduler_factory.return_scheduler(optimizer=optimizer)

        return [optimizer], [{"scheduler": scheduler,
                              "interval": "step",
                              "monitor": "val_mcrmse",
                              "frequency": self.frequency}]

    def freeze_backbone(self):
        self.transformer_model.freeze_backbone()

    def unfreeze_backbone(self):
        self.transformer_model.unfreeze_backbone()
