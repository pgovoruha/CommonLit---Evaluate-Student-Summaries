from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
import codecs
from text_unidecode import unidecode
import re


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


def combine_values(key: str, outputs: List[Dict]) -> np.ndarray:
    return torch.cat([item[key] for item in outputs]).detach().cpu().numpy()


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def preprocess_dataframe(dataframe, text_col_name):

    dataframe.loc[:, text_col_name] = dataframe[text_col_name].map(lambda x: resolve_encodings_and_normalize(x))
    # dataframe.loc[:, text_col_name] = dataframe[text_col_name].map(lambda x: clean_text(x))
    # dataframe.loc[:, text_col_name] = dataframe[text_col_name].map(lambda x: x.replace('\n', '[BR]'))

    return dataframe


def preprocess_data(dataframe):

    dataframe = preprocess_dataframe(dataframe, 'text')
    dataframe = preprocess_dataframe(dataframe, 'prompt_text')
    # dataframe = preprocess_dataframe(dataframe, 'corrected_summary')
    # dataframe = preprocess_dataframe(dataframe, 'corrected_prompts')

    return dataframe
