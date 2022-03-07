from training.datasets.text_classification import AGNewsDataModule
from inference.data_processors import Processor


class TestAGNewsDataModule:

    def test_get_sample(self, tmpdir):
        processor = Processor(preprocessing=[])
        dm = AGNewsDataModule(data_dir=tmpdir, processor=processor)

        next(dm.train_dataloader()._get_iterator())
