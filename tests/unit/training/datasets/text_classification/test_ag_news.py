import nlpiper
from nlpiper.core import Compose

from training.datasets.text_classification import AGNewsDataModule
from inference.data_processors import Processor
from inference.data_processors.transformers.preprocessing import NLPiperIntegration, VocabTransform


class TestAGNewsDataModule:

    def test_get_sample(self, tmpdir):
        dataset = [(0, 'a'), (1, 'b'), (0, 'c'), (2, 'd')]
        vocab = VocabTransform()
        preprocessing = [
            NLPiperIntegration(pipeline=nlpiper.core.Compose([
                nlpiper.transformers.cleaners.CleanPunctuation(),
                nlpiper.transformers.tokenizers.MosesTokenizer()
            ])),
            vocab
        ]
        processor = Processor(preprocessing=preprocessing)

        data_module = AGNewsDataModule(data_dir=tmpdir, processor=processor)

        vocab.build_vocab(processor, dataset)

        next(data_module.train_dataloader()._get_iterator())
