import os

import nlpiper
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from inference.architectures.text_classification import BaselineModel
from training.trainer import TextClassificationTrainer
from training.datasets.text_classification import AGNewsDataModule


if __name__ == "__main__":
    NUMBER_CLASSES = 4
    EMBED_DIM = 100

    model_checkpoint = ModelCheckpoint(monitor="val_CrossEntropyLoss", mode="min", save_weights_only=True)
    early_stop_callback = EarlyStopping(monitor="val_CrossEntropyLoss", mode="min", patience=4)
    mf_logger = MLFlowLogger(
        experiment_name="AG News - Text Classification",
        run_name="Baseline",
        tracking_uri=os.getenv('MLFLOW_URI')
    )

    model = BaselineModel(vocab_size=3e4, embed_dim=EMBED_DIM, num_class=NUMBER_CLASSES)

    data_module = AGNewsDataModule()

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
