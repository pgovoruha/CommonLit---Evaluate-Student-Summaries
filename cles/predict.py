import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from cles.models.models import CustomModel, CustomModelWithPromptText, get_model
from cles.datasets.datasets import CommonLitDataset
from cles.utils import collate, preprocess_data


def get_dataloader(dataframe, tokenizer, cfg, num_workers):

    dataframe = preprocess_data(dataframe)
    dataset = CommonLitDataset(dataframe=dataframe,
                               dataset_type='inference',
                               tokenizer=tokenizer,
                               max_length=cfg.dataset.max_length,
                               target_cols=cfg.dataset.target_cols,
                               include_prompt_text=cfg.dataset.include_prompt_text,
                               use_corrected_text=cfg.dataset.use_corrected_text)

    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=num_workers)

    return dataloader


def predict(model, dataloader, device):

    preds = []
    for batch in tqdm(dataloader):

        batch = collate(batch)
        for k, v in batch.items():
            batch[k] = v.to(device)

        with torch.no_grad():
            prediction = model(batch)
            preds.append(prediction.cpu().numpy())
    predictions = np.concatenate(preds)
    return predictions
