import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import List


class CommonLitDataset(Dataset):

    def __init__(self,
                 dataframe: pd.DataFrame,
                 dataset_type: str,
                 tokenizer: PreTrainedTokenizerBase,
                 target_cols: List[str],
                 max_length: int = 1024):
        self.dataframe = dataframe
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.target_cols = target_cols
        self.max_length = max_length

    def __len__(self):

        return self.dataframe.shape[0]

    def __getitem__(self, idx: int):

        text = self.dataframe.loc[idx, 'text']

        # inputs = self.tokenizer.encode_plus(
        #     text,
        #     return_tensors=None,
        #     add_special_tokens=True,
        #     max_length=self.max_length,
        #     padding="max_length",
        #     truncation=True
        # )

        inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True)

        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)

        if self.dataset_type == 'inference':
            return inputs
        else:
            targets = self.dataframe.loc[idx, self.target_cols].values.astype(float)
            targets = torch.tensor(targets, dtype=torch.float)
            return inputs, targets
