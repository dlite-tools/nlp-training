"""Get max value from Inference"""
from typing import Tuple, Dict

import torch

from inference.data_processors.transformers import BaseTransformer


class GetMaxPrediction(BaseTransformer):
    """Get max value from Inference"""

    def __init__(self, classes: Dict[int, str]):
        """Get prediction from classification model

        Parameters
        ----------
        classes: Dict[int, str]
            Dictionary with classes to be mapped, e.g. `{0: 'class_0', 1: 'class_1'}`
        """
        self.classes = classes

    def __call__(self, inference: torch.Tensor) -> Tuple[str, float]:
        predicted_class = self.classes[inference.argmax(1).item() + 1]
        confidence = round(inference.softmax(1).max().item(), 4) * 100

        return predicted_class, confidence
