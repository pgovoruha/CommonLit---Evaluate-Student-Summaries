import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from typing import List


class CommonLitDataset(Dataset):

    def __init__(self,
                 dataframe: pd.DataFrame,
                 dataset_type: str,
                 tokenizer: PreTrainedTokenizerBase,
                 target_cols: List[str],
                 max_length: int = 1024,
                 ):
        self.dataframe = dataframe
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.target_cols = target_cols
        self.max_length = max_length

    def __len__(self):

        return self.dataframe.shape[0]

    def __getitem__(self, idx: int):

        text = self.dataframe.loc[idx, 'text']
        prompt_question = self.dataframe.loc[idx, 'prompt_question']
        # input_text = prompt_question + self.tokenizer.cls_token + text
        inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                return_tensors=None,
                                add_special_tokens=True)

        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)

        if self.dataset_type == 'inference':
            return inputs
        else:
            targets = self.dataframe.loc[idx, self.target_cols].values.astype(float)
            targets = torch.tensor(targets, dtype=torch.float)
            return inputs, targets


class CommonLitDataset2(Dataset):

    def __init__(self,
                 dataframe: pd.DataFrame,
                 dataset_type: str,
                 tokenizer: PreTrainedTokenizerBase,
                 target_cols: List[str],
                 prompt_text_embeddings: np.ndarray,
                 max_length: int = 1024):
        super().__init__()
        self.dataframe = dataframe
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.target_cols = target_cols
        self.prompt_text_embeddings = prompt_text_embeddings
        self.max_length = max_length

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        text = self.dataframe.loc[idx, 'text']
        prompt_text_embedding = self.prompt_text_embeddings[idx, :]
        inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                return_tensors=None,
                                add_special_tokens=True)

        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        prompt_text_embedding = torch.tensor(prompt_text_embedding, dtype=torch.float)

        if self.dataset_type == 'inference':
            return prompt_text_embedding, inputs
        else:
            targets = self.dataframe.loc[idx, self.target_cols].values.astype(float)
            targets = torch.tensor(targets, dtype=torch.float)
            return prompt_text_embedding, inputs, targets


