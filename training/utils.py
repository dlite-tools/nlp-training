import os
import tempfile

import torch


def log_vocab(vocab, mf_logger):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, 'vocab.pth')
        torch.save(vocab.vocab, temp_file)
        mf_logger.experiment.log_artifact(mf_logger.run_id, temp_file)
