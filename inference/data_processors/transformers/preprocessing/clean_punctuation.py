"""Remove punctuation from text."""
from string import punctuation

from nlpiper.core import Document

from inference.data_processors.transformers import BaseTransformer


class CleanPunctuation(BaseTransformer):
    """Remove punctuation from a document.

    Callable arguments:

    Args:
        doc (Document): document to be cleaned.

    Returns:
        Document without punctuation or None if `inplace=True`.

    Example:
        >>> doc = Document("Document without punctuation!")
        >>> cleaner = CleanPunctuation()
        >>> out = cleaner(doc)
        >>> out.cleaned
        "Document without punctuation"
    """

    def __call__(self, doc: Document) -> Document:
        """Remove punctuation from a document.

        Args:
            doc (Document): document to be cleaned.

        Returns: Document
        """

        doc.cleaned = doc.cleaned.translate(str.maketrans('', '', punctuation))

        return doc
