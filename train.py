import wandb
import os
import dotenv
import torch
import pytorch_lightning as pl
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from commonlit.lightningmodule.modelmodule import LitModel
# torch.set_float32_matmul_precision('medium')


@hydra.main(config_path='config', config_name='config')
def train(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    os.chdir('/home/pavlo/Documents/projects/kaggle/CommonLit---Evaluate-Student-Summaries')
    dotenv.load_dotenv()

    pl.seed_everything(42)
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.datamodule)
    criterion = instantiate(cfg.criterion)
    model.base_transformer.save_pretrained(f'pretrained_models/{cfg.experiment.transformer_name}')
    datamodule.tokenizer.save_pretrained(f'pretrained_models/{cfg.experiment.transformer_name}')
    lit_model = LitModel(
        transformer_model=model,
        criterion=criterion,
        cfg_optimizer=cfg.optimizer,
        cfg_scheduler=cfg.scheduler,
        learning_rate=cfg.learning_rate,
    )

    if cfg.experiment.freeze_backbone:
        lit_model.freeze_backbone()

    checkpoint_callback = ModelCheckpoint(monitor='val_mcrmse',
                                          filename=f'lightning_checkpoints/{cfg.experiment.run_name}',
                                          auto_insert_metric_name=True,
                                          save_top_k=-1
                                          )
    early_stopping_callback = EarlyStopping(monitor='val_mcrmse', patience=5, mode='min')

    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb_logger = WandbLogger(project="kaggle_common_list", name=cfg.experiment.run_name)
    trainer = pl.Trainer(logger=wandb_logger, max_epochs=cfg.experiment.max_epochs,
                         deterministic=True, callbacks=[checkpoint_callback, early_stopping_callback],
                         accumulate_grad_batches=cfg.experiment.gradient_accumulation_steps,
                         val_check_interval=cfg.experiment.val_check_intervals
                         )
    # trainer = pl.Trainer(overfit_batches=1, logger=wandb_logger)
    tuner = pl.tuner.Tuner(trainer)
    tuner.lr_find(lit_model, datamodule=datamodule)
    trainer.validate(model=lit_model, datamodule=datamodule)
    trainer.fit(lit_model, datamodule=datamodule)

    lit_model = LitModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(lit_model, datamodule=datamodule)

    torch.save(lit_model.transformer_model.state_dict(), f'model_weights/{cfg.experiment.run_name}.pth')



if __name__ == '__main__':
    train()