import torch
import torch.nn as nn


class BinaryAlignClassifier(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor):
        return self.fc(hidden_states)