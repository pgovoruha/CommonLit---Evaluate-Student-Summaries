import torch
import torch.nn as nn


class Pooling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, attention_mask):
        raise NotImplementedError


class MeanPooling(Pooling):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, attention_mask):
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MaxPooling(Pooling):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, attention_mask):
        last_hidden_state=outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


class MinPooling(Pooling):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, attention_mask):
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e4
        min_embeddings, _ = torch.min(embeddings, dim=1)
        return min_embeddings


class ConcatenatePooling(Pooling):

    def __init__(self, n_last_layers: int=4):
        super().__init__()
        self.n_last_layers = n_last_layers

    def forward(self, outputs, attention_mask):
        all_hidden_states = torch.stack(outputs[2])

        concatenate_pooling = torch.cat(
            tuple(all_hidden_states[-i] for i in range(1, self.n_last_layers+1)), -1)
        concatenate_pooling = concatenate_pooling[:, 0]

        return concatenate_pooling

