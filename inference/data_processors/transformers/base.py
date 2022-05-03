"""Base transformer Module."""
from abc import (
    ABC,
    abstractmethod
)
from typing import Tuple

import torch


class BaseTransformer(ABC):
    """Class for transformers that deal with documents."""

    _data_aug = False

    @abstractmethod
    def __call__(self, text: str) -> str:
        """Implement method."""
        pass
