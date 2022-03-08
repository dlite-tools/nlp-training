import pytest
import nlpiper
from nlpiper.core import Compose, Document

from inference.data_processors.transformers.preprocessing import NLPiperIntegration


class TestNLPiperIntegration:

    def test_call(self):
        inputs = Document('Hi my name is.')
        expected_output = ['Hi', 'my', 'name', 'is']

        pipeline = Compose([
            nlpiper.transformers.cleaners.CleanPunctuation(),
            nlpiper.transformers.tokenizers.MosesTokenizer(),
        ])

        t = NLPiperIntegration(pipeline=pipeline)

        computed_doc = t(inputs)

        assert all([computed_token.output == expected_token
                    for computed_token, expected_token in zip(computed_doc.tokens, expected_output)])
