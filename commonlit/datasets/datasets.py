import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class CommonLitDataset(Dataset):

    def __init__(self,
                 dataframe: pd.DataFrame,
                 dataset_type: str,
                 tokenizer: PreTrainedTokenizerBase,
                 max_length: int = 1024):
        self.dataframe = dataframe
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):

        return self.dataframe.shape[0]

    def __getitem__(self, idx: int):

        text = self.dataframe.loc[idx, 'text']
        tokenized_text = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True)

        inputs = {
            "input_ids": torch.tensor(tokenized_text['input_ids'], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized_text['attention_mask'], dtype=torch.long)
        }

        if self.dataset_type == 'inference':
            return inputs
        else:
            targets = self.dataframe.loc[idx, ['content', 'wording']].values.astype(float)
            inputs['targets'] = torch.tensor(targets, dtype=torch.float)
            return inputs
