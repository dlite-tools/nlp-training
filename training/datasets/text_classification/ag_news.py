from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchtext.datasets import AG_NEWS
from nlpiper.core import Document

from inference.data_processors.processor import Processor


class AGNewsDataModule(pl.LightningDataModule):
    """Generate data loaders for training, validation and testing."""

    def __init__(self, processor: Processor, data_dir: str = '.data', batch_size: int = 32, num_workers: int = 4):
        """Field recognition DataModule.

        Parameters
        ----------
        processor: Processor

        data_dir: str
            Directory path for the dataset.
        batch_size: int
            Batch size.
        num_workers: int
            Number of workers for the data loader.
        """
        super().__init__()
        self.processor = processor
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.setup()

    def setup(self, stage: Optional[str] = None):
        """Do setup training, validation and test datasets."""
        train_dataset, self.test_dataset = AG_NEWS(data_dir=self.data_dir, split=("train", "test"))
        train_len = int(len(train_dataset) * 0.8)
        val_len = len(train_dataset) - train_len

        self.train_dataset, self.val_dataset = random_split(dataset=train_dataset,
                                                            lengths=[train_len, val_len],
                                                            generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        """Data loader for training data."""
        return DataLoader(self.train_dataset, self.batch_size, num_workers=self.num_workers, shuffle=True,
                          collate_fn=self.generate_batch)

    def val_dataloader(self):
        """Data loader for validation data."""
        return DataLoader(self.val_dataset, self.batch_size, num_workers=self.num_workers,
                          collate_fn=self.generate_batch)

    def test_dataloader(self):
        """Data loader for testing data."""
        return DataLoader(self.test_dataset, self.batch_size, num_workers=self.num_workers,
                          collate_fn=self.generate_batch)

    def generate_batch(self, batch):
        label, text = zip(*batch)
        return torch.stack(label), torch.stack(text)
