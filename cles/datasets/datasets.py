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
                 include_prompt_text: bool = False,
                 use_corrected_text: bool = False
                 ):
        self.dataframe = dataframe
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.target_cols = target_cols
        self.max_length = max_length
        self.include_prompt_text = include_prompt_text
        self.use_corrected_text = use_corrected_text

    def __len__(self):

        return self.dataframe.shape[0]

    def __getitem__(self, idx: int):

        if self.use_corrected_text:
            text = self.dataframe.loc[idx, 'corrected_summary']
            prompt_text = self.dataframe.loc[idx, 'corrected_prompts']
            prompt_question = self.dataframe.loc[idx, 'prompt_question']
            prompt_title = self.dataframe.loc[idx, 'prompt_title']
        else:
            text = self.dataframe.loc[idx, 'text']
            prompt_text = self.dataframe.loc[idx, 'prompt_text']
            prompt_question = self.dataframe.loc[idx, 'prompt_question']
            prompt_title = self.dataframe.loc[idx, 'prompt_title']

        if self.include_prompt_text:
            input_text = text + self.tokenizer.sep_token + prompt_text
        else:
            input_text = text
        inputs = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True,
                                return_tensors=None, add_special_tokens=True
                                )

        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)

        if self.dataset_type == 'inference':
            return inputs
        else:
            targets = self.dataframe.loc[idx, self.target_cols].values.astype(float)
            targets = torch.tensor(targets, dtype=torch.float)
            return inputs, targets


class CommonLitDatasetWithPromptText(CommonLitDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int):

        text = self.dataframe.loc[idx, 'text']
        prompt_question = self.dataframe.loc[idx, 'prompt_question']
        prompt_title = self.dataframe.loc[idx, 'prompt_title']
        prompt_text = self.dataframe.loc[idx, 'prompt_text']

        text_input = prompt_question + self.tokenizer.sep_token + prompt_title + self.tokenizer.sep_token + \
            text

        text_inputs = self.tokenizer(text_input, max_length=self.max_length, padding='max_length', truncation=True,
                                     return_tensors=None,
                                     add_special_tokens=True
                                     )
        prompt_inputs = self.tokenizer(prompt_text, max_length=self.max_length, padding='max_length', truncation=True,
                                       return_tensors=None,
                                       add_special_tokens=True
                                       )

        for k, v in text_inputs.items():
            text_inputs[k] = torch.tensor(v, dtype=torch.long)

        for k, v in prompt_inputs.items():
            prompt_inputs[k] = torch.tensor(v, dtype=torch.long)

        if self.dataset_type == 'inference':
            return prompt_inputs, text_inputs
        else:
            targets = self.dataframe.loc[idx, self.target_cols].values.astype(float)
            targets = torch.tensor(targets, dtype=torch.float)
            return prompt_inputs, text_inputs, targets


def get_dataset(custom_model_name, dataframe: pd.DataFrame,
                dataset_type: str,
                tokenizer: PreTrainedTokenizerBase,
                target_cols: List[str],
                max_length: int = 1024,
                include_prompt_text: bool = False,
                use_corrected_text: bool = False):

    if custom_model_name == 'CustomModel':
        return CommonLitDataset(
            dataframe=dataframe,
            dataset_type=dataset_type,
            tokenizer=tokenizer,
            target_cols=target_cols,
            max_length=max_length,
            include_prompt_text=include_prompt_text,
            use_corrected_text=use_corrected_text
        )
    elif custom_model_name == 'CustomModelWithPromptText':
        return CommonLitDatasetWithPromptText(
            dataframe=dataframe,
            dataset_type=dataset_type,
            tokenizer=tokenizer,
            target_cols=target_cols,
            max_length=max_length,
            include_prompt_text=include_prompt_text,
            use_corrected_text=use_corrected_text
        )
