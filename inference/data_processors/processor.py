"""Processor Module."""
from typing import Sequence

from nlpiper.core import Document

from inference.data_processors.transformers.base import (
    Transformer
)


class Processor:
    """Apply pre-processing and post-processing to samples."""

    train = False

    def __init__(self, preprocessing: Sequence[Transformer]):
        """Apply preprocessing to samples.

        Parameters
        ----------
        preprocessing: Tuple[Transformer]
            List with transforms to be applied on preprocessing.
        """
        self.preprocessing = preprocessing

    def preprocess(self, doc: Document) -> Document:
        """Apply pre-processing to samples.

        Parameters
        ----------
        doc: Document
            Pydantic document object with all document metadata.

        Returns
        -------
         invoice
            Pydantic document object updated with `preprocess` applied.
        """
        for transform in self.preprocessing:
            if transform._data_aug and not self.train:  # skip data augmentation if not in training mode
                continue

            doc = transform(doc)

        return doc
