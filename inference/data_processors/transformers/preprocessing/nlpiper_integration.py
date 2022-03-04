"""NLPiper integration for preprocessing documents."""

from nlpiper.core.document import Document
from nlpiper.core.composition import Compose

from inference.data_processors.transformers.base import Transformer


class NLPiperIntegration(Transformer):
    """NLPiper integration."""

    def __init__(self, pipeline: Compose):
        """Define NLPiper pipeline.

        Parameters
        ----------
        pipeline : Compose
            Pipeline that will apply pre-processing on a document.
        """
        self.pipeline = pipeline

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
        doc = self.pipeline(doc)
        for token in doc.tokens:
            token.output = token.cleaned
        return doc
