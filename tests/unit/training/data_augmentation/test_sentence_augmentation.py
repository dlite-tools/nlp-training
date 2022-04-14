from nlpiper.core import Document

from training.data_augmentation import SentenceAugmentation


class TestSentenceAugmentation:

    def test_sentence_augmentation(self):
        doc = Document('Test this transformation.')
        t = SentenceAugmentation()

        computed_doc = t(doc.copy(deep=True))

        assert computed_doc.cleaned != doc.cleaned
