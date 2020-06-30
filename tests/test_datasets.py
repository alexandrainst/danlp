import shutil
import unittest
from collections import defaultdict
from tempfile import NamedTemporaryFile

from flair.datasets import ColumnCorpus
from pyconll.unit import Conll
from spacy.gold import GoldCorpus

from danlp.datasets import DDT, WikiAnn, DATASETS, DSD, EuroparlSentiment1,EuroparlSentiment2, LccSentiment, TwitterSent
from danlp.datasets.word_sim import WordSim353Da
from danlp.utils import write_simple_ner_dataset, read_simple_ner_dataset


class TestNerDatasets(unittest.TestCase):

    def setUp(self):
        self.train_len = 4383
        self.dev_len = 564
        self.test_len = 565

        self.ddt = DDT()  # Load dataset

    def test_write_and_read_simple_ner_dataset(self):
        sentences = [
            ["Jeg", "gik", "en", "tur", "i", "København"],
            ["Alexandra", "Instituttet", "arbejder", "med", "NLP"]
        ]

        entities = [
            ["O", "O", "O", "O", "O", "B-LOC"],
            ["B-ORG", "I-ORG", "O", "O", "O"]
        ]
        tmp_file = NamedTemporaryFile().name
        write_simple_ner_dataset(sentences, entities, tmp_file)

        loaded_sents, loaded_ents = read_simple_ner_dataset(tmp_file)

        self.assertEqual(sentences, loaded_sents)
        self.assertEqual(entities, loaded_ents)



    def test_ddt_dataset(self):
        train, dev, test = self.ddt.load_as_conllu(predefined_splits=True)

        self.assertIsInstance(train, Conll)
        self.assertIsInstance(dev, Conll)
        self.assertIsInstance(test, Conll)

        self.assertEqual([len(train), len(dev), len(test)], [self.train_len, self.dev_len, self.test_len])

        full_dataset = self.ddt.load_as_conllu(predefined_splits=False)
        self.assertEqual(len(full_dataset), self.train_len + self.dev_len + self.test_len)

    def test_ddt_simple_ner(self):
        train, dev, test = self.ddt.load_as_simple_ner(predefined_splits=True)

        self.assertEqual([len(train[0]), len(dev[0]), len(test[0])], [self.train_len, self.dev_len, self.test_len])

        all_sentences, all_entities = self.ddt.load_as_simple_ner(predefined_splits=False)
        self.assertEqual(len(all_sentences), self.train_len + self.dev_len + self.test_len)

        data = defaultdict(int)
        for entities in train[1]:
            for entity in entities:
                if "B" in entity:
                    data[entity[2:]] += 1
        self.assertDictEqual(data, {'ORG': 802, 'LOC': 945, 'PER': 1249,
                                    'MISC': 1007})

    def test_ddt_dataset_with_flair(self):
        flair_corpus = self.ddt.load_with_flair()

        self.assertIsInstance(flair_corpus, ColumnCorpus)

        flair_lens = [len(flair_corpus.train), len(flair_corpus.dev), len(flair_corpus.test)]
        self.assertEqual(flair_lens, [self.train_len, self.dev_len, self.test_len])

        ner_tags = flair_corpus.make_tag_dictionary('ner').idx2item
        asserted_ner_tags = [
            b'B-ORG', b'B-PER', b'B-LOC', b'B-MISC',
            b'I-ORG', b'I-PER', b'I-LOC', b'I-MISC',
            b'O', b'<START>', b'<STOP>', b'<unk>'
        ]
        self.assertCountEqual(ner_tags, asserted_ner_tags)
# temporary omitted spacy from test
#     def test_ddt_dataset_with_spacy(self):
#         ddt = DDT()  # Load dataset
#         corpus = ddt.load_with_spacy()

#         num_sents_train = 0
#         for paragraph in [paragraph[1] for paragraph in list(corpus.train_tuples)]:
#             num_sents_train += len(paragraph)

#         self.assertIsInstance(corpus, GoldCorpus)
#         self.assertEqual(self.train_len, num_sents_train)

#     def test_wikiann_dataset(self):
#         # Change to a sample of the full wikiann to ease test computation
#         DATASETS['wikiann']['url'] = "https://danlp.s3.eu-central-1.amazonaws.com/test-datasets/da.tar.gz"
#         DATASETS['wikiann']['size'] = 2502
#         DATASETS['wikiann']['md5_checksum'] = 'd0271de38ae23f215b5117450efb9ace'

#         wikiann = WikiAnn()

#         corpus = wikiann.load_with_flair()

#         self.assertEqual([len(corpus.train), len(corpus.dev), len(corpus.test)], [21, 2, 3])

#         ner_tags = corpus.make_tag_dictionary('ner').idx2item
#         asserted_ner_tags = [
#             b'B-ORG', b'B-PER', b'B-LOC',
#             b'I-ORG', b'I-PER', b'I-LOC',
#             b'O', b'<START>', b'<STOP>', b'<unk>'
#         ]
#         self.assertCountEqual(ner_tags, asserted_ner_tags)

#         spacy_gold = wikiann.load_with_spacy()
#         self.assertIsInstance(spacy_gold, GoldCorpus)

#         num_train_sents = len(list(spacy_gold.train_tuples)[0][1])
#         num_dev_sents = len(list(spacy_gold.dev_tuples)[0][1])
#         self.assertEqual(num_dev_sents + num_train_sents, 26)

#         shutil.rmtree(wikiann.dataset_dir)

class TestSimilarityDatasets(unittest.TestCase):
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

class TestSentimentDatasets(unittest.TestCase):
    def test_europarlsentiment1(self):
        eusent = EuroparlSentiment1()
        df = eusent.load_with_pandas()
        self.assertEqual(len(df), 184)
        
    def test_europarlsentiment2(self):
        eusent = EuroparlSentiment2()
        df = eusent.load_with_pandas()
        self.assertEqual(len(df), 957)
        self.assertListEqual(list(df.columns), ['text', 'sub/obj', 'polarity'])
        
    def test_lccsentiment(self):
        sent = LccSentiment()
        df = sent.load_with_pandas()
        self.assertEqual(len(df), 499)
       
       
if __name__ == '__main__':
    unittest.main()

