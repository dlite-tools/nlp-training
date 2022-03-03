"""Abstract Module for text classification basic models."""

from typing import Any, Optional

import pytorch_lightning as pl
import torchmetrics
from torch import optim, nn


class TextClassificationTrainer(pl.LightningModule):
    """Text Classification Abstract Class."""

    def __init__(self, model: nn.Module, num_class: int, loss: Optional[Any] = None) -> None:
        """Initialize Text Classification.

        Args:
            model (nn.Module): Pytorch Model.
            num_class (int): number of classes.
            loss (Optional[Any]): Loss function.
        """
        super().__init__()
        self.num_class = num_class
        self.model = model
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.train_f1 = torchmetrics.F1(num_classes=self.num_class, average="macro")
        self.valid_f1 = torchmetrics.F1(num_classes=self.num_class, average="macro")
        self.test_f1 = torchmetrics.F1(num_classes=self.num_class, average="macro")

        if loss is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = loss

    def forward(self, x, offsets):
        """Model forward pass.

        Args:
            x (torch.Tensor): text given to the model for the forward pass.

        Returns: torch.Tensor
        """
        return self.model(x, offsets)

    def configure_optimizers(self) -> Any:
        """Configure optimizer for pytorch lighting.
        Returns: optimizer for pytorch lighting.
        """
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        return optimizer

    def training_step(self, batch, batch_idx):
        """Get training step.

        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.

        Returns: Tensor
        """
        labels, text, offsets = batch
        output = self.forward(text, offsets)
        loss = self.loss(output, labels)
        self.log('train_loss', loss)
        self.train_acc(output, labels)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.train_f1(output, labels)
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Get validation step.

        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.

        Returns: Tensor
        """
        labels, text, offsets = batch
        output = self.forward(text, offsets)
        loss = self.loss(output, labels)
        self.log('valid_loss', loss, on_epoch=True)
        self.valid_acc(output, labels)
        self.log('valid_acc', self.valid_acc, on_epoch=True)
        self.valid_f1(output, labels)
        self.log('valid_f1', self.valid_f1, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Get test step.

        Args:
            batch (List[Tensor]): Data for training.
            batch_idx (int): batch index.
        Returns: Tensor
        """
        labels, text, offsets = batch
        output = self.forward(text, offsets)
        loss = self.loss(output, labels)
        self.log('test_loss', loss, on_epoch=True)
        self.test_acc(output, labels)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.test_f1(output, labels)
        self.log('test_f1', self.test_f1, on_epoch=True)
        return loss

    def setup_logs(self):
        """Log Optim and Scheduler."""
        optim_configs = self.configure_optimizers()
        self.params["loss"] = self.loss
        if isinstance(optim_configs, tuple):
            self.params["optim"], self.params["scheduler"] = optim_configs
            self.params["optim"] = self.params["optim"][0]
        else:
            self.params["optim"] = optim_configs

        for param in self.params["optim"].param_groups:
            param["params"] = None
