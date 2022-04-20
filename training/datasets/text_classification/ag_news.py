"""AG News Module."""
from typing import Any, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchtext.datasets import AG_NEWS
from torchtext.data.functional import to_map_style_dataset

from inference.data_processors.processor import Processor


class AGNewsDataModule(pl.LightningDataModule):
    """Generate data loaders for training, validation and testing."""

    def __init__(self, processor: Processor, data_dir: str = '.data', batch_size: int = 32, num_workers: int = 8):
        """Field recognition DataModule.

        Parameters
        ----------
        processor: Processor

        data_dir: str
            Directory path for the dataset.
        batch_size: int
            Batch size.
        num_workers: int
            Number of workers for DataLoader.
        """
        super().__init__()
        self.processor = processor
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.setup()

    def setup(self, stage: Optional[str] = None):
        """Do setup training, validation and test datasets."""
        train_iter, test_iter = AG_NEWS(root=self.data_dir, split=("train", "test"))

        train_dataset = to_map_style_dataset(train_iter)
        self.test_dataset = to_map_style_dataset(test_iter)
        train_len = int(len(train_dataset) * 0.8)
        val_len = len(train_dataset) - train_len

        self.train_dataset, self.val_dataset = random_split(dataset=train_dataset,
                                                            lengths=[train_len, val_len],
                                                            generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        """Apply data loader for training data."""
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers,
                          collate_fn=lambda batch: self.generate_batch(batch, True))

    def val_dataloader(self):
        """Apply data loader for validation data."""
        return DataLoader(self.val_dataset, self.batch_size, num_workers=self.num_workers,
                          collate_fn=self.generate_batch)

    def test_dataloader(self):
        """Apply data loader for testing data."""
        return DataLoader(self.test_dataset, self.batch_size, num_workers=self.num_workers,
                          collate_fn=self.generate_batch)

    def generate_batch(self, batch: Any, train: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate batch.

        Parameters
        ----------
        batch : Any
            Batch returned from dataset.
        train : bool
            If `True`, data augmentation will be applied if given to Processor, else will not be.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        self.processor.train = train
        labels, texts = zip(*batch)
        docs = []
        offsets = [0]
        labels = [label - 1 for label in labels]

        for text in texts:
            text = self.processor.preprocess(text)
            offsets.append(len(text))
            docs.extend(text)
        offsets = torch.tensor(offsets[:-1], dtype=torch.long).cumsum(dim=0)
        return torch.tensor(labels, dtype=torch.int64), torch.tensor(docs, dtype=torch.int64), offsets
