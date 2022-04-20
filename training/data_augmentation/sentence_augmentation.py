"""Sentence Augmentation."""

import nltk
import nlpaug.augmenter.word as naw

from inference.data_processors.transformers.base import BaseTransformer


class SentenceAugmentation(BaseTransformer):
    """Randomly change and add similar expression of a word for sentence augmentation."""

    _data_aug = True
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')

    aug = naw.SynonymAug()

    def __call__(self, text: str) -> str:
        return self.aug.augment(text)
