import shutil
import unittest

from pyconll.unit import Conll

from danlp.datasets import DDT, WikiAnn, DATASETS


class TestNerDatasets(unittest.TestCase):

    def test_ddt_dataset(self):
        train_len = 4383
        dev_len = 564
        test_len = 565

        ddt = DDT()  # Load dataset

        train, dev, test = ddt.load_as_conllu(predefined_splits=True)

        self.assertIsInstance(train, Conll)
        self.assertIsInstance(dev, Conll)
        self.assertIsInstance(test, Conll)

        self.assertEqual([len(train), len(dev), len(test)], [train_len, dev_len, test_len])

        full_dataset = ddt.load_as_conllu(predefined_splits=False)
        self.assertEqual(len(full_dataset), train_len+dev_len+test_len)

        flair_corpus = ddt.load_with_flair()
        flair_lens = [len(flair_corpus.train), len(flair_corpus.dev), len(flair_corpus.test)]
        self.assertEqual(flair_lens, [train_len, dev_len, test_len])

        ner_tags = flair_corpus.make_tag_dictionary('ner').idx2item
        asserted_ner_tags = [
            b'B-ORG', b'B-PER', b'B-LOC',
            b'I-ORG', b'I-PER', b'I-LOC',
            b'O', b'<START>', b'<STOP>', b'<unk>'
        ]
        self.assertCountEqual(ner_tags, asserted_ner_tags)

    def test_wikiann_dataset(self):
        # Change to a sample of the full wikiann to ease test computation
        DATASETS['wikiann']['url'] = "https://danlp.s3.eu-central-1.amazonaws.com/test-datasets/da.tar.gz"
        DATASETS['wikiann']['size'] = 2502
        DATASETS['wikiann']['md5_checksum'] = 'd0271de38ae23f215b5117450efb9ace'

        wikiann = WikiAnn()

        corpus = wikiann.load_ner_with_flair()

        self.assertEqual([len(corpus.train), len(corpus.test)], [21, 3])

        ner_tags = corpus.make_tag_dictionary('ner').idx2item
        asserted_ner_tags = [
            b'B-ORG', b'B-PER', b'B-LOC',
            b'I-ORG', b'I-PER', b'I-LOC',
            b'O', b'<START>', b'<STOP>', b'<unk>'
        ]
        self.assertCountEqual(ner_tags, asserted_ner_tags)

        shutil.rmtree(wikiann.dataset_dir)


