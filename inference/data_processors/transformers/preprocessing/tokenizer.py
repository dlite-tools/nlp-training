"""Tokenize text."""
from typing import List

from inference.data_processors.transformers import BaseTransformer


class BasicTokenizer(BaseTransformer):

    def __call__(self, text: str) -> List[str]:
        """Tokenize a document.

        Args:
            text (str): text to be tokenized.

        Returns: List[str]
        """

        return text.split(' ')
