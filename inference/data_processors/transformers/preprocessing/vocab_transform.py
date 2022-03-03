from typing import Iterable, Optional

import torch
from nlpiper.core.document import Document
from torchtext.vocab import Vocab, build_vocab_from_iterator
from tqdm import tqdm

from inference.data_processors.transformers.base import Transformer
from inference.data_processors.processor import Processor


class VocabTransform(Transformer):

    def __init__(self, vocab: Optional[Vocab] = None):
        self.vocab = vocab

    def __call__(self, doc: Document) -> Document:
        if isinstance(self.vocab, Vocab):
            for token in doc.tokens:
                token.output = self.vocab([token.output])[0]
            doc.output = torch.tensor([token.output for token in doc.tokens])
            return doc
        else:
            return doc

    def build_vocab(self, processor: Processor, dataset: Iterable):
        print('Building the Vocab.')

        def yield_tokens(dataset: Iterable):
            for _, text in tqdm(dataset):
                yield [token.output for token in processor.preprocess(Document(text)).tokens]

        self.vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

    def __len__(self):
        return len(self.vocab) if self.vocab is not None else None
