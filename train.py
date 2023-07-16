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


@hydra.main(config_path='config', config_name='config')
def train(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    os.chdir('/home/pavlo/Documents/projects/kaggle/CommonLit---Evaluate-Student-Summaries')
    dotenv.load_dotenv()

    pl.seed_everything(42)
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.datamodule)
    criterion = instantiate(cfg.criterion)

    lit_model = LitModel(
        transformer_model=model,
        criterion=criterion,
        cfg_optimizer=cfg.optimizer,
        cfg_scheduler=cfg.scheduler,
        learning_rate=cfg.learning_rate,
    )

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', filename=cfg.experiment.run_name, save_top_k=1
                                          )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb_logger = WandbLogger(project="kaggle_common_list", name=cfg.experiment.run_name)
    trainer = pl.Trainer(logger=wandb_logger, max_epochs=cfg.experiment.max_epochs,
                         deterministic=True, callbacks=[checkpoint_callback, early_stopping_callback],
                         accumulate_grad_batches=cfg.experiment.gradient_accumulation_steps)
    tuner = pl.tuner.Tuner(trainer)
    tuner.lr_find(lit_model, datamodule=datamodule)
    trainer.fit(lit_model, datamodule=datamodule)
    trainer.test(lit_model, datamodule=datamodule)

    lit_model.load_from_checkpoint(f'kaggle_common_list/{wandb.run.id}/checkpoints/{cfg.experiment.run_name}.ckpt')
    torch.save(lit_model.transformer_model.state_dict(), f'model_weights/{cfg.experiment.run_name}.pth')



if __name__ == '__main__':
    train()