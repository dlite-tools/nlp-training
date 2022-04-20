"""Training script."""
import os
import tempfile

import torch
from torchtext.datasets import AG_NEWS
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor
)

from inference.architectures.text_classification import BaselineModel
from inference.data_processors.transformers import BaseTransformer
from inference.data_processors.processor import Processor
from inference.data_processors.transformers.preprocessing import (
    VocabTransform
)
from training.trainer import TextClassificationTrainer
from training.datasets.text_classification import AGNewsDataModule

if __name__ == "__main__":
    NUMBER_CLASSES = 4
    EMBED_DIM = 64
    BATCH_SIZE = 64
    NUM_WORKERS = 8

    model_checkpoint = ModelCheckpoint(monitor="valid_loss", mode="min", save_weights_only=True)
    early_stop_callback = EarlyStopping(monitor="valid_loss", mode="min", patience=4)
    learning_rate_monitor = LearningRateMonitor()

    mf_logger = MLFlowLogger(
        experiment_name="AG News - Text Classification",
        run_name="Baseline",
    )

    vocab = VocabTransform()

    class Tokenize(BaseTransformer):
        def __call__(self, text):
            return text.split(' ')

    preprocessing = [
        Tokenize(),
        vocab
    ]
    processor = Processor(preprocessing=preprocessing)

    data_module = AGNewsDataModule(processor=processor, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)

    vocab.build_vocab(processor, AG_NEWS(split='train'))
    mf_logger.experiment.log_artifact(mf_logger.run_id, __file__)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, 'vocab.pth')
        torch.save(vocab.vocab, temp_file)
        mf_logger.experiment.log_artifact(mf_logger.run_id, temp_file)

    model = BaselineModel(vocab_size=len(vocab), embed_dim=EMBED_DIM, num_class=NUMBER_CLASSES)

    model_trainer = TextClassificationTrainer(
        model=model,
        num_class=NUMBER_CLASSES
    )

    trainer = Trainer(
        callbacks=[model_checkpoint, early_stop_callback, learning_rate_monitor],
        max_epochs=5,
        logger=mf_logger,
        gpus=torch.cuda.device_count(),
    )

    trainer.fit(model_trainer, data_module)
    trainer.test(datamodule=data_module)
