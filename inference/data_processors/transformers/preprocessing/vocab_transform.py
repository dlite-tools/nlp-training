"""Vocabulary Transform."""

from typing import Iterable, Optional, List

import torch
from torchtext.vocab import Vocab, build_vocab_from_iterator
from tqdm import tqdm

from inference.data_processors.transformers import BaseTransformer
from inference.data_processors.processor import Processor


class VocabTransform(BaseTransformer):
    """Vocabulary transform."""

    def __init__(self, vocab: Optional[Vocab] = None):
        """Vocabulary transform.

        Parameters
        ----------
        vocab : Optional[Vocab]
            Vocabulary
        """
        self.vocab = vocab

    def __call__(self, text: List[str]) -> torch.Tensor:  # type: ignore
        """Apply pipeline on Document.

        Parameters
        ----------
        text : List[str]
            List of token to be preprocessed.

        Returns
        -------
        list of int
        """
        if isinstance(self.vocab, Vocab):
            return torch.tensor([self.vocab([token])[0] for token in text])
        else:
            return text  # type: ignore

    def build_vocab(self, processor: Processor, dataset: Iterable):
        """Build Vocabulary.

        Parameters
        ----------
        processor : Processor
            Processor used for preprocessing data.
        dataset : Iterable
            Iterable which returns a label and sample.
        """
        print('Building the Vocab.')

        def yield_tokens(dataset: Iterable):
            for _, text in tqdm(dataset):
                yield processor.preprocess(text)

        self.vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>", '<pad>', '<eos>'])
        self.vocab.set_default_index(self.vocab["<unk>"])

    def __len__(self):
        """Vocabulary length."""
        return len(self.vocab) if self.vocab is not None else None
