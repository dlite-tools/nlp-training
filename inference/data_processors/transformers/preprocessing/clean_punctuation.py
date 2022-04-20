"""Remove punctuation from text."""
from string import punctuation

from inference.data_processors.transformers import BaseTransformer


class CleanPunctuation(BaseTransformer):

    def __call__(self, text: str) -> str:
        """Remove punctuation from a document.

        Args:
            text (str): text to be cleaned.

        Returns: str
        """

        return text.translate(str.maketrans('', '', punctuation))
