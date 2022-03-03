from nlpiper.core.document import Document
from nlpiper.core.composition import Compose

from inference.data_processors.transformers.base import Transformer


class NLPiperIntegration(Transformer):

    def __init__(self, pipeline: Compose):
        self.pipeline = pipeline

    def __call__(self, doc: Document) -> Document:
        doc = self.pipeline(doc)
        for token in doc.tokens:
            token.output = token.cleaned
        return doc
