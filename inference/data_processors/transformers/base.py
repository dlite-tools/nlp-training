"""Base transformer Module."""
from abc import (
    ABC,
    abstractmethod
)

from nlpiper.core import Document


class BaseTransformer(ABC):
    """Class for transformers that deal with documents."""

    _data_aug = False

    @abstractmethod
    def __call__(self, doc: Document) -> Document:
        """Implement method."""
        pass
