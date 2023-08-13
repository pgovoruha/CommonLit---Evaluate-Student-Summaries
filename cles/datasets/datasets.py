import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import List
from sentence_transformers import SentenceTransformer


class CommonLitDataset(Dataset):

    def __init__(self,
                 dataframe: pd.DataFrame,
                 dataset_type: str,
                 tokenizer: PreTrainedTokenizerBase,
                 sentence_transformer_tokenizer: PreTrainedTokenizerBase,
                 target_cols: List[str],
                 max_length: int = 1024,
                 ):
        self.dataframe = dataframe
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.target_cols = target_cols
        self.max_length = max_length
        self.sentence_transformer_tokenizer = sentence_transformer_tokenizer

    def __len__(self):

        return self.dataframe.shape[0]

    def __getitem__(self, idx: int):

        text = self.dataframe.loc[idx, 'text']
        prompt_text = self.dataframe.loc[idx, 'prompt_text']
        prompt_title = self.dataframe.loc[idx, 'prompt_title']
        prompt_question = self.dataframe.loc[idx, 'prompt_question']
        # input_text = prompt_question + self.tokenizer.sep_token + text
        prompt_input = prompt_question + self.sentence_transformer_tokenizer.sep_token + \
            prompt_title + self.sentence_transformer_tokenizer.sep_token + prompt_text
        inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                return_tensors=None,
                                add_special_tokens=True)

        prompt_inputs = self.sentence_transformer_tokenizer(prompt_input,
                                                            add_special_tokens=True,
                                                            max_length=self.max_length, padding='max_length',
                                                            truncation=True)

        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)

        for k, v in prompt_inputs.items():
            prompt_inputs[k] = torch.tensor(v, dtype=torch.long)

        if self.dataset_type == 'inference':
            return prompt_inputs, inputs
        else:
            targets = self.dataframe.loc[idx, self.target_cols].values.astype(float)
            targets = torch.tensor(targets, dtype=torch.float)
            return prompt_inputs, inputs, targets
