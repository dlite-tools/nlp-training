"""NLPiper integration for preprocessing documents."""
from typing import Any

from nlpiper.core.document import Document
from nlpiper.core.composition import Compose

from inference.data_processors.transformers import BaseTransformer


class NLPiperIntegration(BaseTransformer):
    """NLPiper integration."""

    def __init__(self, pipeline: Compose):
        """Define NLPiper pipeline.

        Parameters
        ----------
        pipeline : Compose
            Pipeline that will apply pre-processing on a document.
        """
        self.pipeline = pipeline

    def __call__(self, text: str) -> Any:
        """Apply pipeline on Document.

        Parameters
        ----------
        text : str
            text to be preprocessed.

        Returns
        -------
        Preprocessed text.
        """
        doc = self.pipeline(Document(text))
        if doc.tokens is not None:
            for token in doc.tokens:
                token.output = token.cleaned

            doc.output = [token.output for token in doc.tokens]
        else:
            doc.output = doc.cleaned
        return doc.output
