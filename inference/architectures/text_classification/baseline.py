"""Baseline Model."""

import torch
from torch import nn


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
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, x: torch.Tensor, offsets: torch.Tensor):
        """Model inference.

        Parameters
        ----------
        x : torch.Tensor
            Sample.
        offsets : torch.Tensor
            Document offsets.

        Returns
        -------
        torch.Tensor
        """
        embedded = self.embedding(x, offsets)
        return self.fc(embedded)
