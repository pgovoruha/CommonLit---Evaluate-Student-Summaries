import torch
import torch.nn as nn


class CustomHead(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.liner = torch.nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.liner(x)
        return x


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim, num_targets, v_output_shape=1, dropout_1=0.1, dropout_2=0.1):
        super().__init__()
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, v_output_shape)

        self.dropout_1 = nn.Dropout(dropout_1)
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)

        self.dropout_2 = nn.Dropout(dropout_2)
        self.linear_2 = nn.Linear(hidden_dim, num_targets)

    def aggregate_state(self, outputs, attention_mask):

        att = torch.tanh(self.W(outputs))

        score = self.V(att)
        score.masked_fill_((attention_mask.unsqueeze(-1) == 0), -float("inf"))

        attention_weights = torch.softmax(score, dim=1)

        context_vector = attention_weights * outputs

        return torch.sum(context_vector, dim=1)

    def forward(self, inputs, attention_mask):

        aggregated_state = self.aggregate_state(inputs, attention_mask)
        x_1 = self.dropout_1(aggregated_state)
        x_1 = self.linear_1(x_1)
        x_1 = torch.relu(x_1)

        x = aggregated_state + x_1
        x = self.dropout_2(x)
        x = self.linear_2(x)

        return x
