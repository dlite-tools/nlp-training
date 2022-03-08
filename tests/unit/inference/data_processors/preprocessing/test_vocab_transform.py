import torch
import nlpiper
from nlpiper.core import Document, Compose

from inference.data_processors import Processor
from inference.data_processors.transformers.preprocessing import VocabTransform, NLPiperIntegration


class TestVocabTransform:

    def test_build_vocab(self):
        dataset = [(0, 'a'), (1, 'b'), (0, 'c'), (2, 'd')]
        p = Processor([NLPiperIntegration(pipeline=Compose([
            nlpiper.transformers.tokenizers.MosesTokenizer()
        ]))])

        v = VocabTransform()
        v.build_vocab(p, dataset)

        tokens = ['a', 'b', 'c', 'd', 'e']
        expected_token_map = [3, 4, 5, 6, 0]
        assert v.vocab(tokens) == expected_token_map

    def test_call(self):
        dataset = [(0, 'a'), (1, 'b'), (0, 'c'), (2, 'd')]
        vocab = VocabTransform()
        p = Processor([NLPiperIntegration(pipeline=Compose([
            nlpiper.transformers.tokenizers.MosesTokenizer()])),
            vocab
        ])

        vocab.build_vocab(p, dataset)

        computed_doc = p.preprocess(Document('a b c d e'))

        torch.testing.assert_close(computed_doc.output, torch.tensor([3, 4, 5, 6, 0, 2]))
