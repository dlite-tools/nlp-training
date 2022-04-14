"""Baseline Model."""

import torch
from torch import nn
from torch.nn import functional as F


class BaselineModel(nn.Module):
    """Baseline Model."""

    def __init__(self, vocab_size: int, embed_dim: int, num_class: int):
        """Baseline Model.

        Parameters
        ----------
        vocab_size : int
            Vocabulary size.
        embed_dim : int
            Embedding size.
        num_class : int
            Number of classes.
        """
        super(BaselineModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, x: torch.Tensor):
        """Model inference.

        Parameters
        ----------
        x : torch.Tensor
            Sample.

        Returns
        -------
        torch.Tensor
        """
        embedded = self.embedding(x)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        return self.fc(pooled)
