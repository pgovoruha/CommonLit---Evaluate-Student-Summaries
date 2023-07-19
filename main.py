import wandb
import os
import dotenv
import torch
import pytorch_lightning as pl
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from commonlit.lightningmodule.modelmodule import LitModel
from commonlit.models.models import CustomModel
from commonlit.factories.factories import (PoolFactory, HeadFactory, OptimizerFactory, SchedulerFactory,
                                           CriterionFactory)
from commonlit.lightningmodule.datamodule import LitCommonLitDataset
torch.set_float32_matmul_precision('medium')


@hydra.main(config_path='config', config_name='config')
def train(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    os.chdir('/home/pavlo/Documents/projects/kaggle/CommonLit---Evaluate-Student-Summaries')
    dotenv.load_dotenv()
    pl.seed_everything(42)
    pool_factory = PoolFactory(cfg=cfg)
    head_factory = HeadFactory(cfg=cfg)
    criterion_factory = CriterionFactory(cfg=cfg)
    optimizer_factory = OptimizerFactory(cfg=cfg)
    scheduler_factory = SchedulerFactory(cfg=cfg)

    model = CustomModel(cfg=cfg,
                        pool_factory=pool_factory,
                        head_factory=head_factory)
    print(model)

    datamodule = LitCommonLitDataset(
        train_path=cfg.data.train_path,
        test_path=cfg.data.test_path,
        val_path=cfg.data.val_path,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        tokenizer_name=cfg.base_transformer,
        target_cols=cfg.target_cols
    )

    model.config.save_pretrained(f'models_arc/{cfg.base_transformer}')
    datamodule.tokenizer.save_pretrained(f'models_arc/{cfg.base_transformer}')
    lit_model = LitModel(
        transformer_model=model,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        criterion_factory=criterion_factory,
        learning_rate=cfg.learning_rate,
        frequency=cfg.val_check_intervals
    )

    if cfg.freeze_backbone:
        lit_model.freeze_backbone()

    checkpoint_callback = ModelCheckpoint(monitor='val_mcrmse',
                                          filename=f'lightning_checkpoints/{cfg.run_name}',
                                          auto_insert_metric_name=True,
                                          save_top_k=5
                                          )
    early_stopping_callback = EarlyStopping(monitor='val_mcrmse', patience=5, mode='min')

    if cfg.logger == 'tensorboard':
        logger = TensorBoardLogger(save_dir='tensorboard_logs', name=cfg.run_name)
    else:
        wandb.login(key=os.getenv('WANDB_API_KEY'))
        logger = WandbLogger(project="kaggle_common_list", name=cfg.run_name)
    trainer = pl.Trainer(logger=logger, max_epochs=cfg.max_epochs,
                         deterministic=True, callbacks=[checkpoint_callback, early_stopping_callback],
                         accumulate_grad_batches=cfg.gradient_accumulation_steps,
                         val_check_interval=cfg.val_check_intervals
                         )
    # trainer = pl.Trainer(overfit_batches=1, logger=wandb_logger)
    tuner = pl.tuner.Tuner(trainer)
    tuner.lr_find(lit_model, datamodule=datamodule)
    trainer.validate(model=lit_model, datamodule=datamodule)
    trainer.fit(lit_model, datamodule=datamodule)

    lit_model = LitModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(lit_model, datamodule=datamodule)

    torch.save(lit_model.transformer_model.state_dict(), f'model_weights/{cfg.run_name}.pth')


if __name__ == '__main__':
    train()