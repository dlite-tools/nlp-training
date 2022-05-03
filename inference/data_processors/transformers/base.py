"""Base transformer Module."""
from abc import (
    ABC,
    abstractmethod
)


class BaseTransformer(ABC):
    """Class for transformers that deal with documents."""

    _data_aug = False

    @abstractmethod
    def __call__(self, text: str) -> str:
        """Implement method."""
        pass
