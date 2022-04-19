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
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, x: torch.Tensor, offsets: torch.Tensor):
        """Model inference.

        Parameters
        ----------
        x : torch.Tensor
            Sample.
        offsets : torch.Tensor
            Offsets.

        Returns
        -------
        torch.Tensor
        """
        embedded = self.embedding(x, offsets)
        return self.fc(embedded)

    def load_from_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint into memory.

        Parameters
        ----------
        checkpoint_path : str
            Checkpoint path
        """
        weights = torch.load(checkpoint_path, map_location='cpu')
        if any('model.' in layer for layer in list(weights['state_dict'].keys())):
            weights = {key.replace('model.', ''): value for (key, value) in weights['state_dict'].items()}
        else:
            weights = weights['state_dict']

        self.load_state_dict(weights)  # type:ignore
