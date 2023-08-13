import wandb
import os
import dotenv
import torch
import torch.utils.checkpoint
import lightning as L
from lightning.pytorch.tuner.tuning import Tuner
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import warnings

from cles.lightningmodule.modelmodule import LitModel
from cles.models.models import CustomModel
from cles.factories.factories import (PoolFactory, HeadFactory, OptimizerFactory, SchedulerFactory,
                                      CriterionFactory)
from cles.lightningmodule.datamodule import LitCommonLitDataset

torch.set_float32_matmul_precision('medium')
warnings.filterwarnings('ignore')


@hydra.main(config_path='config', config_name='config')
def train(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    os.chdir('/home/pavlo/Documents/projects/kaggle/CommonLit---Evaluate-Student-Summaries')
    dotenv.load_dotenv()
    L.seed_everything(cfg.seed)
    pool_factory = PoolFactory(cfg=cfg)
    head_factory = HeadFactory(cfg=cfg)
    criterion_factory = CriterionFactory(cfg=cfg)
    optimizer_factory = OptimizerFactory(cfg=cfg)
    scheduler_factory = SchedulerFactory(cfg=cfg)

    model = CustomModel(cfg=cfg,
                        pool_factory=pool_factory,
                        head_factory=head_factory)
    print(model)
    model.sentence_transformer.save_pretrained(f'{cfg.models_conf}/{cfg.group}/sentence_transformer')
    datamodule = LitCommonLitDataset(
        train_path=cfg.data.train_path,
        test_path=cfg.data.test_path,
        val_path=cfg.data.val_path,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        tokenizer_name=cfg.base_transformer,
        sentence_transformer=cfg.sentence_transformer,
        target_cols=cfg.target_cols
    )

    model.config.save_pretrained(f'{cfg.models_conf}/{cfg.group}/{cfg.base_transformer}')
    datamodule.tokenizer.save_pretrained(f'{cfg.models_conf}/{cfg.group}/{cfg.base_transformer}')
    datamodule.sentence_transformer_tokenizer.save_pretrained(f'{cfg.models_conf}/{cfg.group}/sentence_transformer')
    lit_model = LitModel(
        transformer_model=model,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        criterion_factory=criterion_factory,
        learning_rate=cfg.learning_rate,
        frequency=cfg.val_check_intervals
    )

    if cfg.enable_gradient_checkpoint:
        lit_model.transformer_model.backbone.gradient_checkpointing_enable()

    if cfg.freeze_embeddings:
        lit_model.freeze_embeddings()

    # lit_model = torch.compile(lit_model)

    checkpoint_callback = ModelCheckpoint(monitor='val_mcrmse',
                                          filename=f'lightning_checkpoints/{cfg.run_name}',
                                          auto_insert_metric_name=True,
                                          save_top_k=5
                                          )
    early_stopping_callback = EarlyStopping(monitor='val_mcrmse', patience=cfg.patience, mode='min',
                                            min_delta=0.001)

    wandb.login(key=os.getenv('WANDB_API_KEY'))
    logger = WandbLogger(project="kaggle_common_eval_summaries", name=cfg.run_name,
                         group=cfg.group)

    trainer = L.Trainer(logger=logger, max_epochs=cfg.max_epochs,
                        deterministic=True,
                        callbacks=[checkpoint_callback, early_stopping_callback],
                        val_check_interval=cfg.val_check_intervals,
                        accumulate_grad_batches=cfg.gradient_accumulation_steps,
                        num_sanity_val_steps=2,
                        gradient_clip_val=cfg.gradient_clip_val,
                        min_epochs=1)

    # tuner = Tuner(trainer)
    # tuner.lr_find(lit_model, datamodule=datamodule, min_lr=2e-8, max_lr=1e-3, num_training=1000)

    trainer.fit(lit_model, datamodule=datamodule)

    lit_model = LitModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(lit_model, datamodule=datamodule)

    os.makedirs(f'model_weights/{cfg.group}', exist_ok=True)
    torch.save(lit_model.transformer_model.state_dict(), f'{cfg.model_weights}/{cfg.group}/{cfg.run_name}.pth')

    with open(f"{cfg.models_conf}/{cfg.group}/train_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    train()
