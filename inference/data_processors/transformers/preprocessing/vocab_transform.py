"""Vocabulary Transform."""

from typing import Iterable, Optional

import torch
from nlpiper.core.document import Document
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

    def __call__(self, doc: Document) -> Document:
        """Apply pipeline on Document.

        Parameters
        ----------
        doc : Document
        Pydantic document object with all document metadata.

        Returns
        -------
        Document
        """
        if isinstance(self.vocab, Vocab):
            for token in doc.tokens:
                token.output = self.vocab([token.output])[0]
            doc.output = torch.cat((torch.tensor([token.output for token in doc.tokens]), torch.tensor([2])))
            return doc
        else:
            return doc

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
                yield [token.output for token in processor.preprocess(Document(text)).tokens]

        self.vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>", '<pad>', '<eos>'])
        self.vocab.set_default_index(self.vocab["<unk>"])

    def __len__(self):
        """Vocabulary length."""
        return len(self.vocab) if self.vocab is not None else None
