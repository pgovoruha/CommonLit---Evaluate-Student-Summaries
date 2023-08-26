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
    def __init__(self):
        super().__init__()

    def forward(self, outputs, inputs):
        # last_hidden_state = outputs[0]
        last_hidden_state = outputs
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # input_mask_expanded = attention_mask.expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MaxPooling(Pooling):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, inputs):
        last_hidden_state = outputs[0]
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


class MinPooling(Pooling):
    def __init__(self):
        super(MinPooling, self).__init__()

    def forward(self, outputs, inputs):
        last_hidden_state = outputs[0]
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e4
        min_embeddings, _ = torch.min(embeddings, dim=1)
        return min_embeddings


class ConcatenatePooling(Pooling):

    def __init__(self, n_last_layers: int = 4):
        super().__init__()
        self.n_last_layers = n_last_layers

    def forward(self, outputs, inputs):
        all_hidden_states = torch.stack(outputs.hidden_states)

        concatenate_pooling = torch.cat(
            tuple(all_hidden_states[-i] for i in range(1, self.n_last_layers+1)), -1)
        concatenate_pooling = concatenate_pooling[:, 0]

        return concatenate_pooling


class MeanMaxPooling(Pooling):

    def __init__(self):
        super().__init__()
        self.mean_pool = MeanPooling()
        self.max_pool = MaxPooling()

    def forward(self, outputs, inputs):
        x1 = self.mean_pool(outputs, inputs)
        x2 = self.max_pool(outputs, inputs)

        return torch.cat([x1, x2], dim=1)


class Conv1DPool(Pooling):

    def __init__(self, backbone_config, middle_dim: int = 256):
        super().__init__()
        self.backbone_config = backbone_config
        self.cnn1 = nn.Conv1d(self.backbone_config.hidden_size, middle_dim, kernel_size=2, padding=1)

    def forward(self, outputs, inputs):
        last_hidden_state = outputs[0].permute(0, 2, 1)
        out = self.cnn1(last_hidden_state)

        return out


class CLSPooling(Pooling):

    def __init__(self, layer_index: int = 11):
        super().__init__()
        self.layer_index = layer_index

    def forward(self, outputs, inputs):

        all_hidden_states = torch.stack(outputs[2])
        cls_embeddings = all_hidden_states[self.layer_index + 1, :, 0]

        return cls_embeddings


class WeightedLayerPooling(Pooling):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, outputs, inputs):
        all_hidden_states = torch.stack(outputs.hidden_states)
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        weighted_average = weighted_average[:, 0]
        return weighted_average


class LSTMPooler(Pooling):

    def __init__(self, num_layers, hidden_size, hidden_dim_lstm):
        super().__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden_dim_lstm = hidden_dim_lstm
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_dim_lstm, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, outputs, inputs):

        all_hidden_states = torch.stack(outputs[2])

        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:,  -1, :])
        return out


class AttentionPooling(Pooling):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, outputs, inputs):
        last_hidden_states = outputs[0]
        attention_mask = inputs['attention_mask']
        weights_mask = attention_mask.unsqueeze(-1)
        att = torch.tanh(self.W(last_hidden_states))
        score = self.V(att)
        score[attention_mask == 0] = -1e4
        attention_weights = torch.softmax(score, dim=1)
        context_vector = torch.sum(attention_weights * weights_mask * last_hidden_states, dim=1)
        return context_vector


class TwoAttentionPools(Pooling):

    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.attn1 = AttentionPooling(in_features, hidden_dim)
        self.attn2 = AttentionPooling(in_features, hidden_dim)

    def forward(self, outputs, inputs):
        x1 = self.attn1(outputs, inputs)
        x2 = self.attn2(outputs, inputs)
        return x1, x2


class GeMPooling(Pooling):
    def __init__(self, dim=1, p=3, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, outputs, inputs):
        outputs = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.shape)
        outputs = ((outputs.clamp(min=self.eps) * attention_mask_expanded).pow(self.p)).sum(self.dim)
        ret = (outputs / (attention_mask_expanded.sum(self.dim))).clip(min=self.eps)
        ret = ret.pow(1 / self.p)
        return ret


class SimpleAttentionPooling(Pooling):

    def __init__(self, in_features: int, middle_features: int):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(in_features, middle_features),
            nn.Tanh(),
            nn.Linear(middle_features, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, outputs, inputs):
        last_hidden_state = outputs[0]
        weights = self.attention(last_hidden_state)
        context_vector = torch.sum(weights * last_hidden_state, dim=1)
        return context_vector


class WKPooling(Pooling):
    def __init__(self, layer_start: int = 4, context_window_size: int = 2):
        super(WKPooling, self).__init__()
        self.layer_start = layer_start
        self.context_window_size = context_window_size

    def forward(self, outputs, inputs):
        all_hidden_states = torch.stack(outputs[2])
        ft_all_layers = all_hidden_states
        org_device = ft_all_layers.device
        all_layer_embedding = ft_all_layers.transpose(1, 0)
        all_layer_embedding = all_layer_embedding[:, self.layer_start:, :, :]  # Start from 4th layers output

        # torch.qr is slow on GPU (see https://github.com/pytorch/pytorch/issues/22573). So compute it on CPU until issue is fixed
        all_layer_embedding = all_layer_embedding.cpu()

        attention_mask = inputs['attention_mask'].cpu().numpy()
        unmask_num = np.array([sum(mask) for mask in attention_mask]) - 1  # Not considering the last item
        embedding = []

        # One sentence at a time
        for sent_index in range(len(unmask_num)):
            sentence_feature = all_layer_embedding[sent_index, :, :unmask_num[sent_index], :]
            one_sentence_embedding = []
            # Process each token
            for token_index in range(sentence_feature.shape[1]):
                token_feature = sentence_feature[:, token_index, :]
                # 'Unified Word Representation'
                token_embedding = self.unify_token(token_feature)
                one_sentence_embedding.append(token_embedding)

            # features.update({'sentence_embedding': features['cls_token_embeddings']})

            one_sentence_embedding = torch.stack(one_sentence_embedding)
            sentence_embedding = self.unify_sentence(sentence_feature, one_sentence_embedding)
            embedding.append(sentence_embedding)

        output_vector = torch.stack(embedding).to(org_device)
        return output_vector

    def unify_token(self, token_feature):

        window_size = self.context_window_size

        alpha_alignment = torch.zeros(token_feature.size()[0], device=token_feature.device)
        alpha_novelty = torch.zeros(token_feature.size()[0], device=token_feature.device)

        for k in range(token_feature.size()[0]):
            left_window = token_feature[k - window_size:k, :]
            right_window = token_feature[k + 1:k + window_size + 1, :]
            window_matrix = torch.cat([left_window, right_window, token_feature[k, :][None, :]])
            Q, R = torch.qr(window_matrix.T)

            r = R[:, -1]
            alpha_alignment[k] = torch.mean(self.norm_vector(R[:-1, :-1], dim=0), dim=1).matmul(
                R[:-1, -1]) / torch.norm(r[:-1])
            alpha_alignment[k] = 1 / (alpha_alignment[k] * window_matrix.size()[0] * 2)
            alpha_novelty[k] = torch.abs(r[-1]) / torch.norm(r)

        # Sum Norm
        alpha_alignment = alpha_alignment / torch.sum(alpha_alignment)  # Normalization Choice
        alpha_novelty = alpha_novelty / torch.sum(alpha_novelty)

        alpha = alpha_novelty + alpha_alignment
        alpha = alpha / torch.sum(alpha)  # Normalize

        out_embedding = torch.mv(token_feature.t(), alpha)
        return out_embedding

    def norm_vector(self, vec, p=2, dim=0):

        vec_norm = torch.norm(vec, p=p, dim=dim)
        return vec.div(vec_norm.expand_as(vec))

    def unify_sentence(self, sentence_feature, one_sentence_embedding):

        sent_len = one_sentence_embedding.size()[0]

        var_token = torch.zeros(sent_len, device=one_sentence_embedding.device)
        for token_index in range(sent_len):
            token_feature = sentence_feature[:, token_index, :]
            sim_map = self.cosine_similarity_torch(token_feature)
            var_token[token_index] = torch.var(sim_map.diagonal(-1))

        var_token = var_token / torch.sum(var_token)
        sentence_embedding = torch.mv(one_sentence_embedding.t(), var_token)

        return sentence_embedding

    def cosine_similarity_torch(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)





