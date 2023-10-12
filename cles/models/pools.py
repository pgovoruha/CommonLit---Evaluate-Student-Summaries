import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Pooling(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, inputs):
        raise NotImplementedError


class MeanPooling(Pooling):
    def __init__(self, backbone_config, pool_config):
        super().__init__()
        self.n_last_layers = pool_config.params.n_last_layers
        self.output_dim = backbone_config.hidden_size if self.n_last_layers is None else \
            backbone_config.hidden_size * self.n_last_layers

    def forward(self, outputs, inputs):

        if self.n_last_layers is None:
            last_hidden_state = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states
            last_hidden_state = torch.cat(hidden_states[-self.n_last_layers:], dim=-1)

        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class AttentionPooling(Pooling):
    def __init__(self, backbone_config, pool_config):
        super(AttentionPooling, self).__init__()
        self.num_hidden_layers = backbone_config.num_hidden_layers
        self.hidden_size = backbone_config.hidden_size
        self.hidden_dim_fc = pool_config.params.hidden_dim_fc
        self.dropout = nn.Dropout(pool_config.params.dropout)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.tensor(q_t, requires_grad=True, dtype=torch.float32))

        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hidden_dim_fc))
        self.w_h = nn.Parameter(torch.tensor(w_ht, requires_grad=True, dtype=torch.float32))

        self.output_dim = self.hidden_dim_fc

    def forward(self, outputs, inputs):
        all_hidden_states = outputs.hidden_states

        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out = self.attention(hidden_states)
        out = self.dropout(out)
        return out

    def attention(self, h):
        v = torch.matmul(self.q, h.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), h).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v


class LSTMPooling(Pooling):
    def __init__(self, backbone_config, pooling_config):
        super(LSTMPooling, self).__init__()

        self.num_hidden_layers = backbone_config.num_hidden_layers
        self.hidden_size = backbone_config.hidden_size
        self.hidden_lstm_size = pooling_config.params.hidden_size
        self.dropout_rate = pooling_config.params.dropout_rate
        self.bidirectional = pooling_config.params.bidirectional

        self.is_lstm = pooling_config.params.is_lstm
        self.output_dim = pooling_config.params.hidden_size * 2 if self.bidirectional else \
            pooling_config.params.hidden_size

        if self.is_lstm:
            self.lstm = nn.LSTM(self.hidden_size,
                                self.hidden_lstm_size,
                                bidirectional=self.bidirectional,
                                batch_first=True)
        else:
            self.lstm = nn.GRU(self.hidden_size,
                               self.hidden_lstm_size,
                               bidirectional=self.bidirectional,
                               batch_first=True)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, outputs, inputs):
        all_hidden_states = torch.stack(outputs.hidden_states)

        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out


class MaxPooling(Pooling):
    def __init__(self, backbone_config, pooling_config):
        super(MaxPooling, self).__init__()
        self.output_dim = backbone_config.hidden_size

    def forward(self, outputs, inputs):
        attention_mask = inputs["attention_mask"]
        last_hidden_state = outputs.last_hidden_state

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(last_hidden_state, dim=1)
        return max_embeddings


class MinPooling(Pooling):
    def __init__(self, backbone_config, pooling_config):
        super(MinPooling, self).__init__()
        self.output_dim = backbone_config.hidden_size

    def forward(self, outputs, inputs):
        attention_mask = inputs["attention_mask"]
        last_hidden_state = outputs.last_hidden_state

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        last_hidden_state[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(last_hidden_state, dim=1)
        return min_embeddings


class MinMaxPooling(Pooling):

    def __init__(self, backbone_config, pooling_config):
        super().__init__()
        self.min_pool = MinPooling(backbone_config, pooling_config)
        self.max_pool = MaxPooling(backbone_config, pooling_config)
        self.output_dim = self.min_pool.output_dim + self.max_pool.output_dim

    def forward(self, outputs, inputs):
        y1 = self.min_pool(outputs, inputs)
        y2 = self.max_pool(outputs, inputs)
        y = torch.cat([y1, y2], dim=1)
        return y


class MeanMaxPooling(Pooling):

    def __init__(self, backbone_config, pooling_config):
        super().__init__()
        self.mean_pool = MeanPooling(backbone_config, pooling_config)
        self.max_pool = MaxPooling(backbone_config, pooling_config)
        self.output_dim = self.mean_pool.output_dim + self.max_pool.output_dim

    def forward(self, outputs, inputs):
        y1 = self.mean_pool(outputs, inputs)
        y2 = self.max_pool(outputs, inputs)
        y = torch.cat([y1, y2], dim=1)
        return y


class WeightedLayerPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(WeightedLayerPooling, self).__init__()

        self.num_hidden_layers = backbone_config.num_hidden_layers
        self.layer_start = pooling_config.params.layer_start
        self.layer_weights = pooling_config.params.layer_weights if pooling_config.params.layer_weights is not None else \
            nn.Parameter(torch.tensor([1] * (self.num_hidden_layers + 1 - self.layer_start), dtype=torch.float))

        self.output_dim = backbone_config.hidden_size

    def forward(self, outputs, inputs):
        all_hidden_states = outputs.hidden_states

        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average[:, 0]


class ConcatenatePooling(Pooling):
    def __init__(self, backbone_config, pooling_config):
        super().__init__()

        self.n_layers = pooling_config.params.n_layers
        self.output_dim = backbone_config.hidden_size*self.n_layers

    def forward(self, outputs, inputs):
        all_hidden_states = outputs.hidden_states

        concatenate_pooling = torch.cat([all_hidden_states[-(i + 1)] for i in range(self.n_layers)], -1)
        concatenate_pooling = concatenate_pooling[:, 0]
        return concatenate_pooling


class GeMText(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(GeMText, self).__init__()

        self.dim = int(pooling_config.params.dim)
        self.eps = float(pooling_config.params.eps)
        self.feat_mult = 1

        self.p = nn.Parameter(torch.ones(1) * int(pooling_config.params.p))

        self.output_dim = backbone_config.hidden_size

    def forward(self, outputs, inputs):
        attention_mask = inputs['attention_mask']
        x = outputs.last_hidden_state

        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(x.shape)
        x = (x.clamp(min=self.eps) * attention_mask_expanded).pow(self.p).sum(self.dim)
        ret = x / attention_mask_expanded.sum(self.dim).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret


def create_pooling(backbone_config, pool_config):
    if pool_config.name == 'MeanPooling':
        return MeanPooling(backbone_config, pool_config)
    elif pool_config.name == 'AttentionPooling':
        return AttentionPooling(backbone_config, pool_config)
    elif pool_config.name == 'LSTMPooling':
        return LSTMPooling(backbone_config, pool_config)
    elif pool_config.name == 'GeMText':
        return GeMText(backbone_config, pool_config)
    elif pool_config.name == 'ConcatenatePooling':
        return ConcatenatePooling(backbone_config, pool_config)
    elif pool_config.name == "MaxPooling":
        return MaxPooling(backbone_config, pool_config)
    elif pool_config.name == "MinPooling":
        return MinPooling(backbone_config, pool_config)
    elif pool_config.name == 'MinMaxPooling':
        return MinMaxPooling(backbone_config, pool_config)
    elif pool_config.name == 'MeanMaxPooling':
        return MeanMaxPooling(backbone_config, pool_config)
    elif pool_config.name == 'WeightedLayerPooling':
        return WeightedLayerPooling(backbone_config, pool_config)
    else:
        raise ValueError(f'Unknown pooling {pool_config.name}')
