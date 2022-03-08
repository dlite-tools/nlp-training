import pytest

from inference.data_processors.transformers.base import BaseTransformer


class TestBaseTransformer:

    def test_base_transform(self):
        with pytest.raises(TypeError):
            BaseTransformer()
