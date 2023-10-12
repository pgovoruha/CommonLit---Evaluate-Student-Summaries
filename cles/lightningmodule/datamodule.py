import pandas as pd
import lightning as L
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import multiprocessing
from typing import List, Tuple
from cles.datasets.datasets import CommonLitDataset, CommonLitDatasetWithPromptText, get_dataset
from cles.utils import preprocess_data


class LitCommonLitDataset(L.LightningDataModule):
    def __init__(self,
                 custom_model_name: str,
                 train_path: str,
                 val_path: str,
                 test_path: str,
                 max_length: int,
                 tokenizer_name: str,
                 batch_size: int,
                 target_cols: List[str],
                 include_prompt_text: bool = False,
                 use_corrected_text: bool = False):
        super().__init__()
        self.custom_model_name = custom_model_name
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.target_cols = target_cols
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.include_prompt_text = include_prompt_text
        self.use_corrected_text = use_corrected_text
        if 'gpt' in tokenizer_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # self.tokenizer.add_special_tokens({"additional_special_tokens": ['[P]', '[BR]']})
        self.batch_size = batch_size
        self.num_cpus = multiprocessing.cpu_count()

    def setup(self, stage=None):
        train_df = preprocess_data(pd.read_csv(self.train_path))
        val_df = preprocess_data(pd.read_csv(self.val_path))
        test_df = preprocess_data(pd.read_csv(self.test_path))

        self.train_dataset = get_dataset(custom_model_name=self.custom_model_name,
                                         dataframe=train_df,
                                         dataset_type='train',
                                         tokenizer=self.tokenizer,
                                         max_length=self.max_length,
                                         target_cols=self.target_cols,
                                         include_prompt_text=self.include_prompt_text,
                                         use_corrected_text=self.use_corrected_text)

        self.val_dataset = get_dataset(custom_model_name=self.custom_model_name,
                                       dataframe=val_df,
                                       dataset_type='val',
                                       tokenizer=self.tokenizer,
                                       max_length=self.max_length,
                                       target_cols=self.target_cols,
                                       include_prompt_text=self.include_prompt_text,
                                       use_corrected_text=self.use_corrected_text)
        self.test_dataset = get_dataset(custom_model_name=self.custom_model_name,
                                        dataframe=test_df,
                                        dataset_type='val',
                                        tokenizer=self.tokenizer,
                                        max_length=self.max_length,
                                        target_cols=self.target_cols,
                                        include_prompt_text=self.include_prompt_text,
                                        use_corrected_text=self.use_corrected_text)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_cpus)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_cpus)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_cpus)