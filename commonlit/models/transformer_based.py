import torch
import torch.nn as nn
from transformers import AutoModel


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class TransformerWithCustomHead(torch.nn.Module):

    def __init__(self, base_transformer: str, head: torch.nn.Module):

        super().__init__()
        self.base_transformer = AutoModel.from_pretrained(base_transformer)
        self.head = head
        self.pool = MeanPooling()

    def forward(self, input_ids, attention_mask):
        outputs = self.base_transformer(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.pool(outputs[0], attention_mask)
        y = self.head(outputs)
        return y
