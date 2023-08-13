import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from cles.models.models import CustomModel
from cles.factories.factories import PoolFactory, HeadFactory
from cles.datasets.datasets import CommonLitDataset


def get_model(cfg):

    pool_factory = PoolFactory(cfg=cfg)
    head_factory = HeadFactory(cfg=cfg)

    if cfg.config_path is None:
        raise ValueError('Config path to model should be provided in order to make predictions')
    model = CustomModel(cfg=cfg,
                        pool_factory=pool_factory,
                        head_factory=head_factory)

    return model


def get_dataloader(dataframe, tokenizer, sentence_transformer_tokenizer, cfg, num_workers):

    dataset = CommonLitDataset(dataframe=dataframe,
                               tokenizer=tokenizer,
                               target_cols=cfg.target_cols,
                               sentence_transformer_tokenizer=sentence_transformer_tokenizer,
                               max_length=cfg.max_length,
                               dataset_type='inference')

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers)

    return dataloader


def predict(model, dataloader, device):

    preds = []
    for batch in tqdm(dataloader):
        prompt_inputs, inputs = batch

        for k, v in prompt_inputs.items():
            prompt_inputs[k] = v.to(device)

        for k, v in inputs.items():
            inputs[k] = v.to(device)

        with torch.no_grad():
            prediction = model(prompt_inputs, inputs)
            preds.append(prediction.cpu().numpy())
    predictions = np.concatenate(preds)
    return predictions
