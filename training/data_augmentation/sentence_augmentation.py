"""Sentence Augmentation."""

import nltk
import nlpaug.augmenter.word as naw
from nlpiper.core import Document

from inference.data_processors.transformers.base import Transformer


class SentenceAugmentation(Transformer):
    """Randomly change and add similar expression of a word for sentence augmentation."""

    _data_aug = True
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')

    aug = naw.SynonymAug()

    def __call__(self, doc: Document) -> Document:
        """Sentence Augmentation.

        Parameters
        ----------
        doc : Document
            Pydantic document object with all document metadata.

        Returns
        -------
        Document
        """
        doc.cleaned = self.aug.augment(doc.cleaned)
        return doc
