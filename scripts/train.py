import os

import nlpiper
import torch
from torchtext.datasets import AG_NEWS
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from inference.architectures.text_classification import BaselineModel
from inference.data_processors.processor import Processor
from inference.data_processors.transformers.preprocessing import (
    NLPiperIntegration,
    VocabTransform
)
from training.data_augmentation import SentenceAugmentation
from training.trainer import TextClassificationTrainer
from training.datasets.text_classification import AGNewsDataModule

if __name__ == "__main__":
    NUMBER_CLASSES = 4
    EMBED_DIM = 100

    model_checkpoint = ModelCheckpoint(monitor="valid_loss", mode="min", save_weights_only=True)
    early_stop_callback = EarlyStopping(monitor="valid_loss", mode="min", patience=4)
    mf_logger = MLFlowLogger(
        experiment_name="AG News - Text Classification",
        run_name="Baseline",
        tracking_uri=os.getenv('MLFLOW_URI')
    )

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
        callbacks=[model_checkpoint, early_stop_callback],
        max_epochs=30,
        logger=mf_logger,
        gpus=torch.cuda.device_count()
    )

    trainer.fit(model_trainer, data_module)
    trainer.test(dataloaders=data_module.test_dataloader())
