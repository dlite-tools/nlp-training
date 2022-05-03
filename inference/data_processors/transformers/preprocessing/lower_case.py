"""Lower case text."""
from inference.data_processors.transformers import BaseTransformer


class LowerCase(BaseTransformer):

    def __call__(self, text: str) -> str:
        """Lower case text.

        Args:
            text (str): text to be cleaned.

        Returns: str
        """

        return text.lower()
