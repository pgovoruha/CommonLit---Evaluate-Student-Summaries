import pandas as pd
import lightning as L
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import multiprocessing
from typing import List
from sentence_transformers import SentenceTransformer
from cles.datasets.datasets import CommonLitDataset, CommonLitDataset2


class LitCommonLitDataset(L.LightningDataModule):
    def __init__(self,
                 train_path: str,
                 val_path: str,
                 test_path: str,
                 max_length: int,
                 tokenizer_name: str,
                 batch_size: int,
                 target_cols: List[str]):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.target_cols = target_cols
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.num_cpus = multiprocessing.cpu_count()

    def setup(self, stage=None):
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)

        self.train_dataset = CommonLitDataset(dataframe=train_df,
                                              dataset_type='train',
                                              tokenizer=self.tokenizer,
                                              max_length=self.max_length,
                                              target_cols=self.target_cols)

        self.val_dataset = CommonLitDataset(dataframe=val_df,
                                            dataset_type='val',
                                            tokenizer=self.tokenizer,
                                            max_length=self.max_length,
                                            target_cols=self.target_cols)
        self.test_dataset = CommonLitDataset(dataframe=test_df,
                                             dataset_type='val',
                                             tokenizer=self.tokenizer,
                                             max_length=self.max_length,
                                             target_cols=self.target_cols)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_cpus)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_cpus)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_cpus)


class LitCommonLitDataset2(L.LightningDataModule):
    def __init__(self,
                 train_path: str,
                 val_path: str,
                 test_path: str,
                 max_length: int,
                 tokenizer_name: str,
                 batch_size: int,
                 target_cols: List[str],
                 sentence_model: str):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.target_cols = target_cols
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.sentence_model = sentence_model
        self.num_cpus = multiprocessing.cpu_count()

    def prepare_embeddings(self, dataframe):
        sentence_transformer_model = SentenceTransformer(model_name_or_path=self.sentence_model)
        prompt_embeddings = sentence_transformer_model.encode(dataframe['prompt_text'].tolist())
        return prompt_embeddings

    def setup(self, stage=None):
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)

        train_prompt_embeddings = self.prepare_embeddings(train_df)
        val_prompt_embeddings = self.prepare_embeddings(val_df)
        test_prompt_embeddings = self.prepare_embeddings(test_df)

        self.train_dataset = CommonLitDataset2(dataframe=train_df,
                                               dataset_type='train',
                                               tokenizer=self.tokenizer,
                                               max_length=self.max_length,
                                               target_cols=self.target_cols,
                                               prompt_text_embeddings=train_prompt_embeddings)

        self.val_dataset = CommonLitDataset2(dataframe=val_df,
                                             dataset_type='val',
                                             tokenizer=self.tokenizer,
                                             max_length=self.max_length,
                                             target_cols=self.target_cols,
                                             prompt_text_embeddings=val_prompt_embeddings)
        self.test_dataset = CommonLitDataset2(dataframe=test_df,
                                              dataset_type='val',
                                              tokenizer=self.tokenizer,
                                              max_length=self.max_length,
                                              target_cols=self.target_cols,
                                              prompt_text_embeddings=test_prompt_embeddings)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_cpus)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_cpus)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_cpus)
