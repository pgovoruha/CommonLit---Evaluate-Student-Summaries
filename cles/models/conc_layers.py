import torch
import torch.nn as nn


class ConcatModel(nn.Module):
    def __init__(self, hidden_size, feature_size):
        super(ConcatModel, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size

    def forward(self, feature_vector, outputs):
        last_hidden_state = outputs.last_hidden_state
        batch_size, sequence_length, _ = last_hidden_state.size()
        expanded_feature = feature_vector.unsqueeze(1).expand(-1, sequence_length, -1)
        combined = torch.cat([last_hidden_state, expanded_feature], dim=-1)

        return combined


class AddModel(nn.Module):
    def __init__(self, hidden_size, feature_size):
        super(AddModel, self).__init__()
        self.projection = nn.Linear(feature_size, hidden_size)

    def forward(self, feature_vector, outputs):
        last_hidden_state = outputs.last_hidden_state
        projected_feature = self.projection(feature_vector)
        expanded_feature = projected_feature.unsqueeze(1).expand_as(last_hidden_state)
        combined = last_hidden_state + expanded_feature

        output = self.classifier(combined.mean(dim=1))
        return output


class AttentionModel(nn.Module):
    def __init__(self, hidden_size, feature_size):
        super(AttentionModel, self).__init__()
        self.query_projection = nn.Linear(feature_size, hidden_size)

    def forward(self, feature_vector, outputs):

        last_hidden_state = outputs.last_hidden_state
        query = self.query_projection(feature_vector).unsqueeze(1)  # shape: (batch_size, 1, hidden_size)
        attention_weights = (query * last_hidden_state).sum(dim=-1, keepdim=True)
        attention_weights = nn.functional.softmax(attention_weights, dim=1)

        attended_values = (attention_weights * last_hidden_state).sum(dim=1)

        return attended_values
