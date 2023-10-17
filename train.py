import wandb
import os
import dotenv
import torch
import torch.utils.checkpoint
import lightning as L
from argparse import ArgumentParser
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import warnings
import yaml

from cles.lightningmodule.modelmodule import get_lit_model
from cles.models.models import get_model
from cles.lightningmodule.datamodule import LitCommonLitDataset

warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser.add_argument("--config_path", dest="config_path", default="config/db_large_lstm_pooling.yaml")
parser.add_argument("--fold_id", dest='fold_id', default='3b9047')
parser.add_argument('--model_path', dest='model_path', default=None)

dotenv.load_dotenv()


def get_config(config_path: str):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    config = DictConfig(config)
    return config


def train():

    args = parser.parse_args()
    config_path = args.config_path
    fold_id = args.fold_id
    model_path = args.model_path
    config = get_config(config_path)
    print(OmegaConf.to_yaml(config))
    L.seed_everything(int(config.train.seed))

    model = get_model(config)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    backbone_name = config.backbone.name
    backbone_name = backbone_name.replace('/', '_').replace('-', '_')
    run_name = f'{backbone_name}_{fold_id}'
    train_path = f'{config.root_folder}/{fold_id}/train.csv'
    val_path = f'{config.root_folder}/{fold_id}/test.csv'

    datamodule = LitCommonLitDataset(
        custom_model_name=config.train.model_name,
        train_path=train_path,
        test_path=val_path,
        val_path=val_path,
        batch_size=config.train.batch_size,
        max_length=config.dataset.max_length,
        tokenizer_name=config.backbone.name,
        target_cols=config.dataset.target_cols,
        include_prompt_text=config.dataset.include_prompt_text,
        use_corrected_text = config.dataset.use_corrected_text
    )

    if 'gpt' in config.backbone.name:
        model.backbone.resize_token_embeddings(len(datamodule.tokenizer))
    model.backbone_config.save_pretrained(f'models_config/{config.group_name}/{config.backbone.name}')
    datamodule.tokenizer.save_pretrained(f'models_config/{config.group_name}/{config.backbone.name}')

    checkpoint_callback = ModelCheckpoint(monitor='val_mcrmse',
                                          filename=f'lightning_checkpoints/{config.group_name}/{run_name}',
                                          auto_insert_metric_name=True,
                                          save_top_k=1
                                          )

    early_stopping_callback = EarlyStopping(monitor='val_mcrmse', patience=config.train.patience, mode='min',
                                            min_delta=0.001)

    wandb.login(key=os.getenv('WANDB_API_KEY'))
    logger = WandbLogger(project="kaggle_common_eval_summaries", name=run_name,
                         group=config.group_name)

    lit_model = get_lit_model(config, model)

    trainer = L.Trainer(logger=logger, max_epochs=config.train.max_epochs,
                        deterministic=True,
                        callbacks=[checkpoint_callback, early_stopping_callback],
                        val_check_interval=int(config.train.val_check_intervals),
                        accumulate_grad_batches=int(config.train.gradient_accumulation_steps),
                        num_sanity_val_steps=2,
                        gradient_clip_val=int(config.train.gradient_clip_val),
                        #precision=16,
                        min_epochs=int(config.train.min_epochs)
                        )

    trainer.fit(lit_model, datamodule=datamodule)
    lit_model = lit_model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(lit_model, datamodule=datamodule)
    os.makedirs(f'model_weights/{config.group_name}', exist_ok=True)

    torch.save(lit_model.transformer_model.state_dict(),
               f'model_weights/{config.group_name}/{run_name}.pth')

    torch.cuda.empty_cache()


if __name__ == '__main__':
    train()
