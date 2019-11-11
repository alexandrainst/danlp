import shutil
import unittest

from flair.datasets import ColumnCorpus
from pyconll.unit import Conll
from spacy.gold import GoldCorpus

from danlp.datasets import DDT, WikiAnn, DATASETS, DSD
from danlp.datasets.word_sim import WordSim353Da


class TestNerDatasets(unittest.TestCase):

    def setUp(self):
        self.train_len = 4383
        self.dev_len = 564
        self.test_len = 565

        self.ddt = DDT()  # Load dataset

    def test_ddt_dataset(self):
        train, dev, test = self.ddt.load_as_conllu(predefined_splits=True)

        self.assertIsInstance(train, Conll)
        self.assertIsInstance(dev, Conll)
        self.assertIsInstance(test, Conll)

        self.assertEqual([len(train), len(dev), len(test)], [self.train_len, self.dev_len, self.test_len])

        full_dataset = self.ddt.load_as_conllu(predefined_splits=False)
        self.assertEqual(len(full_dataset), self.train_len + self.dev_len + self.test_len)

    def test_ddt_dataset_with_flair(self):
        flair_corpus = self.ddt.load_with_flair()

        self.assertIsInstance(flair_corpus, ColumnCorpus)

        flair_lens = [len(flair_corpus.train), len(flair_corpus.dev), len(flair_corpus.test)]
        self.assertEqual(flair_lens, [self.train_len, self.dev_len, self.test_len])

        ner_tags = flair_corpus.make_tag_dictionary('ner').idx2item
        asserted_ner_tags = [
            b'B-ORG', b'B-PER', b'B-LOC',
            b'I-ORG', b'I-PER', b'I-LOC',
            b'O', b'<START>', b'<STOP>', b'<unk>'
        ]
        self.assertCountEqual(ner_tags, asserted_ner_tags)

    def test_ddt_dataset_with_spacy(self):
        ddt = DDT()  # Load dataset
        corpus = ddt.load_with_spacy()
        self.assertIsInstance(corpus, GoldCorpus)

    def test_wikiann_dataset(self):
        # Change to a sample of the full wikiann to ease test computation
        DATASETS['wikiann']['url'] = "https://danlp.s3.eu-central-1.amazonaws.com/test-datasets/da.tar.gz"
        DATASETS['wikiann']['size'] = 2502
        DATASETS['wikiann']['md5_checksum'] = 'd0271de38ae23f215b5117450efb9ace'

        wikiann = WikiAnn()

        corpus = wikiann.load_with_flair()

        self.assertEqual([len(corpus.train), len(corpus.dev), len(corpus.test)], [21, 2, 3])

        ner_tags = corpus.make_tag_dictionary('ner').idx2item
        asserted_ner_tags = [
            b'B-ORG', b'B-PER', b'B-LOC',
            b'I-ORG', b'I-PER', b'I-LOC',
            b'O', b'<START>', b'<STOP>', b'<unk>'
        ]
        self.assertCountEqual(ner_tags, asserted_ner_tags)

        spacy_gold = wikiann.load_with_spacy()
        self.assertIsInstance(spacy_gold, GoldCorpus)

        num_train_sents = len(list(spacy_gold.train_tuples)[0][1])
        num_dev_sents = len(list(spacy_gold.dev_tuples)[0][1])
        self.assertEqual(num_dev_sents + num_train_sents, 26)

        shutil.rmtree(wikiann.dataset_dir)

    def test_wordsim353(self):
        ws353 = WordSim353Da()
        df = ws353.load_with_pandas()

        self.assertEqual(len(df), 353)
        self.assertListEqual(list(df.columns), ['da1', 'da2', 'Human (mean)'])
        self.assertEqual(len(ws353.words()), 424)

    def test_dsd(self):
        dsd = DSD()
        df = dsd.load_with_pandas()

        self.assertEqual(len(df), 99)
        self.assertListEqual(list(df.columns), ['word1', 'word2', 'similarity'])
        self.assertEqual(len(dsd.words()), 197)
