import torch
import torch.nn as nn


class PairwiseScorer(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()

        self.fc = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        x (torch.Tensor)
            Pairwise features from query source token (q) expanded (B, H) -> (B, K, K, H)
            and symmetric combination of target token pairs (t_i, t_j) -> (B, K, K, H)

            e.g., x = [q ; t_i + t_j ; t_i (*) t_j] -> (B, K, K, 3H)
                - where feat_dim = 3H
        """
        return self.fc(x)