import nlpiper
import torch
from torchtext.datasets import AG_NEWS
from pytorch_lightning import Trainer

from inference.architectures.text_classification import BaselineModel
from inference.data_processors.processor import Processor
from inference.data_processors.transformers.preprocessing import (
    NLPiperIntegration,
    VocabTransform
)
from training.data_augmentation import SentenceAugmentation
from training.trainer import TextClassificationTrainer
from training.datasets.text_classification import AGNewsDataModule


# TODO: reduce dataset size otherwise the train will be too long
class TestTextClassificationTrainer:

    def test_train(self):
        NUMBER_CLASSES = 4
        EMBED_DIM = 100

        vocab = VocabTransform()
        preprocessing = [
            SentenceAugmentation(),
            NLPiperIntegration(pipeline=nlpiper.core.Compose([
                nlpiper.transformers.cleaners.CleanPunctuation(),
                nlpiper.transformers.tokenizers.MosesTokenizer()
            ])),
            vocab
        ]
        processor = Processor(preprocessing=preprocessing)

        data_module = AGNewsDataModule(processor=processor)

        vocab.build_vocab(processor, AG_NEWS(split='train'))
        model = BaselineModel(vocab_size=len(vocab), embed_dim=EMBED_DIM, num_class=NUMBER_CLASSES)

        model_trainer = TextClassificationTrainer(
            model=model,
            num_class=NUMBER_CLASSES
        )

        trainer = Trainer(
            max_epochs=1,
            logger=None,
            gpus=torch.cuda.device_count(),
            overfit_batches=1
        )

        trainer.fit(model_trainer, datamodule=data_module)
        trainer.test(datamodule=data_module)
