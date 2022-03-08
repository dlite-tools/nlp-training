import pytest
import torch

from inference.architectures.text_classification import BaselineModel


class TestBaselineModel:

    @pytest.mark.parametrize('vocab_size,embed_dim,num_class', [
        (100, 50, 4),
        (100, 5, 10),
    ])
    def test_baseline_model(self, vocab_size, embed_dim, num_class):
        BATCH_SIZE = 3
        MAX_LENGTH = 10
        model = BaselineModel(vocab_size, embed_dim, num_class)

        inputs = torch.randint(0, vocab_size, (BATCH_SIZE, MAX_LENGTH))

        output = model(inputs)
        assert output.shape == (BATCH_SIZE, num_class)
