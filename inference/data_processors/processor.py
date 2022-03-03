"""Processor Module."""
from typing import Sequence, Iterable, Optional

import torch
from nlpiper.core import Document
from torchtext.vocab import build_vocab_from_iterator, Vocab

from inference.data_processors.transformers.base import (
    Transformer
)


class Processor:
    """Apply pre-processing and post-processing to samples."""

    train = False

    def __init__(self, preprocessing: Sequence[Transformer], vocab: Optional[Vocab] = None):
        """Apply preprocessing to samples.

        Parameters
        ----------
        preprocessing: Tuple[Transformer]
            List with transforms to be applied on preprocessing.
        """
        self.preprocessing = preprocessing
        self._vocab = vocab

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

    def build_vocab(self, dataset: Iterable):
        def yield_tokens(dataset: Iterable):
            for _, text in dataset:
                yield [token.cleaned for token in self.preprocess(Document(text)).tokens]

        self._vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])

    def vocab(self, doc: Document):
        return torch.tensor(self._vocab([token.cleaned for token in doc.tokens]))
