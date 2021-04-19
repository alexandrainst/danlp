import shutil
import unittest
from collections import defaultdict
from tempfile import NamedTemporaryFile

from flair.datasets import ColumnCorpus
from pyconll.unit import Conll
from spacy.gold import GoldCorpus

from danlp.datasets import DDT, WikiAnn, DATASETS, DSD, EuroparlSentiment1,EuroparlSentiment2, LccSentiment, TwitterSent, Dacoref, DanNet, DKHate, DaUnimorph
from danlp.datasets.word_sim import WordSim353Da
from danlp.utils import write_simple_ner_dataset, read_simple_ner_dataset

DANLP_STORAGE_URL = 'http://danlp-downloads.alexandra.dk'

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

    def test_ddt_dataset_with_spacy(self):
        ddt = DDT()  # Load dataset
        corpus = ddt.load_with_spacy()

        num_sents_train = 0
        for paragraph in [paragraph[1] for paragraph in list(corpus.train_tuples)]:
            num_sents_train += len(paragraph)

        self.assertIsInstance(corpus, GoldCorpus)
        self.assertEqual(self.train_len, num_sents_train)
        
    def test_wikiann_dataset(self):
        # Change to a sample of the full wikiann to ease test computation
        DATASETS['wikiann']['url'] = DANLP_STORAGE_URL+ "/tests/da.tar.gz"
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
        
class TestCorefDatasets(unittest.TestCase):
    def test_dacoreg(self):
        dacoref = Dacoref() 
        corpus = dacoref.load_as_conllu(predefined_splits=True) 
        self.assertEqual(len(corpus), 3)
        self.assertEqual(len(corpus[0])+len(corpus[1])+len(corpus[2]), 3403)
        self.assertEqual(corpus[0][0][0]['form'], 'På')
        
class TestHateSpeechDatasets(unittest.TestCase):
    def test_dkhate(self):
        dkhate = DKHate() 
        test, train = dkhate.load_with_pandas()
        self.assertEqual(len(test), 329)
        self.assertEqual(len(train), 2960)
        self.assertEqual(set(test['subtask_a'].to_list()), {'NOT', 'OFF'})

class TestDannetDataset(unittest.TestCase):
    def test_dannet(self):
        dannet = DanNet() 
        corpus = dannet.load_with_pandas()
        self.assertEqual(len(corpus), 4)
        self.assertEqual(dannet.synonyms('kat'), ['missekat', 'mis'])
        self.assertEqual(dannet.hypernyms('myre', pos=['Noun']), ['årevingede insekter'])
        self.assertEqual(dannet.hyponyms('myre', pos='Noun'), ['hærmyre', 'skovmyre', 'pissemyre', 'tissemyre'])
        self.assertEqual(dannet.domains('myre', pos='Noun'), ['zoologi'])
        self.assertEqual(dannet.pos('myre'), ['Noun'])
        self.assertEqual(dannet.meanings('myre'), ['ca. 1 cm langt, årevinget insekt med en kraftig in ... (Brug: "Myrer på terrassen, og andre steder udendørs, kan hurtigt blive meget generende")'])
        self.assertEqual(dannet.wordnet_relations('kat'), {'domain', 'has_holo_part', 'eq_has_synonym', 'has_hyperonym', 'role_agent', 'used_for', 'has_mero_part'})
        self.assertEqual(dannet.wordnet_relations('kat', eurowordnet=False), {'domain', 'partMeronymOf', 'hyponymOf', 'partHolonymOf', 'usedFor', 'eqSynonymOf', 'roleAgent'})
        self.assertEqual(dannet._word_from_id(11025614), ['kat'])
        self.assertEqual(dannet._synset_from_id(3264), {'missekat', 'kat,1', 'mis'})

class TestUnimorphDataset(unittest.TestCase):
    def test_unimorph(self):
        unimorph = DaUnimorph() 
        database = unimorph.load_with_pandas()
        self.assertEqual(len(database), 25503)
        self.assertEqual(unimorph.get_inflections('svar'), ['svaredes', 'svarede', 'svarer', 'svares', 'svare', 'svar'])
        self.assertEqual(unimorph.get_inflections('trolde', pos='V'), ['troldedes', 'troldede', 'trolder', 'troldes', 'trolde', 'trold'])
        self.assertEqual(unimorph.get_inflections('trolde', pos='N', with_features=True)[0], {'lemma': 'trold', 'form': 'troldene', 'feats': 'N;DEF;NOM;PL', 'pos': 'N'})
        self.assertEqual(unimorph.get_lemmas('papiret', with_features=True), [{'lemma': 'papir', 'form': 'papiret', 'feats': 'N;DEF;NOM;SG', 'pos': 'N'}])

if __name__ == '__main__':
    unittest.main()