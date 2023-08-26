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

from cles.lightningmodule.modelmodule import LitModel, LitModel2
from cles.models.models import CustomModel, CustomModel2
from cles.factories.factories import (PoolFactory, HeadFactory, OptimizerFactory, SchedulerFactory,
                                      CriterionFactory, CombineFeaturesFactory)
from cles.lightningmodule.datamodule import LitCommonLitDataset, LitCommonLitDataset2

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
    combine_features_factory = CombineFeaturesFactory(cfg=cfg)

    model = CustomModel2(cfg=cfg,
                         pool_factory=pool_factory,
                         head_factory=head_factory,
                         combine_features_factory=combine_features_factory)
    if cfg.enable_gradient_checkpoint:
        print('Enabling gradient checkpointing')
        model.backbone.gradient_checkpointing_enable()
    print(model)
    datamodule = LitCommonLitDataset2(
        train_path=cfg.data.train_path,
        test_path=cfg.data.test_path,
        val_path=cfg.data.val_path,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        tokenizer_name=cfg.base_transformer,
        target_cols=cfg.target_cols,
        sentence_model=cfg.sentence_model
    )

    model.config.save_pretrained(f'{cfg.models_conf}/{cfg.group}/{cfg.base_transformer}')
    datamodule.tokenizer.save_pretrained(f'{cfg.models_conf}/{cfg.group}/{cfg.base_transformer}')
    lit_model = LitModel2(
        transformer_model=model,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        criterion_factory=criterion_factory,
        learning_rate=cfg.learning_rate,
        frequency=cfg.val_check_intervals
    )

    if cfg.freeze_embeddings:
        lit_model.freeze_embeddings()

    if cfg.freeze_n_layers:
        lit_model.freeze_n_layers(n=cfg.number_to_freeze)

    checkpoint_callback = ModelCheckpoint(monitor='val_mcrmse',
                                          filename=f'lightning_checkpoints/{cfg.run_name}',
                                          auto_insert_metric_name=True,
                                          save_top_k=1
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
                        # precision=16,
                        min_epochs=1)

    # tuner = Tuner(trainer)
    # tuner.lr_find(lit_model, datamodule=datamodule, min_lr=2e-8, max_lr=1e-3, num_training=1000)

    trainer.fit(lit_model, datamodule=datamodule)

    lit_model = LitModel2.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(lit_model, datamodule=datamodule)

    os.makedirs(f'model_weights/{cfg.group}', exist_ok=True)
    torch.save(lit_model.transformer_model.state_dict(), f'{cfg.model_weights}/{cfg.group}/{cfg.run_name}.pth')

    with open(f"{cfg.models_conf}/{cfg.group}/train_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    train()
