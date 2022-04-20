"""Processor Module."""
from typing import Any, Sequence

from inference.data_processors.transformers.base import (
    BaseTransformer
)


class Processor:
    """Apply pre-processing to samples."""

    train = False

    def __init__(self, preprocessing: Sequence[BaseTransformer]):
        """Apply preprocessing to samples.

        Parameters
        ----------
        preprocessing: Tuple[Transformer]
            List with transforms to be applied on preprocessing.
        """
        self.preprocessing = preprocessing

    def preprocess(self, text: str) -> Any:
        """Apply pre-processing to samples.

        Parameters
        ----------
        text: str
            Sample to be preprocessed.

        Returns
        -------
         str
            Preprocessed sample.
        """
        for transform in self.preprocessing:
            if transform._data_aug and not self.train:  # skip data augmentation if not in training mode
                continue

            text = transform(text)

        return text
