import unittest

from flair.data import Sentence

from danlp.download import download_model, DEFAULT_CACHE_DIR, _unzip_process_func
from danlp.models import load_flair_ner_model
from danlp.models import load_flair_pos_model

class TestNerTaggers(unittest.TestCase):
    def test_flair_tagger(self):
        # Download model beforehand
        download_model('flair.ner', DEFAULT_CACHE_DIR, process_func=_unzip_process_func, verbose=True)
        print("Downloaded the flair model")

        # Load the NER tagger using the DaNLP wrapper
        flair_model = load_flair_ner_model()

        # Using the flair POS tagger
        sentence = Sentence('jeg hopper på en bil som er rød sammen med Jens-Peter E. Hansen')
        flair_model.predict(sentence)

        expected_string = "jeg hopper på en bil som er rød sammen med Jens-Peter <B-PER> E. <I-PER> Hansen <I-PER>"

        self.assertEqual(sentence.to_tagged_string(), expected_string)

class TestPosTaggers(unittest.TestCase):
    def test_flair_tagger(self):
        # Download model beforehand
        download_model('flair.pos', DEFAULT_CACHE_DIR, process_func=_unzip_process_func, verbose=True)
        print("Downloaded the flair model")

        # Load the POS tagger using the DaNLP wrapper
        flair_model = load_flair_pos_model()

        # Using the flair POS tagger
        sentence = Sentence('jeg hopper på en bil som er rød sammen med Jens-Peter E. Hansen')
        flair_model.predict(sentence)

        expected_string = "jeg <PRON> hopper <VERB> på <ADP> en <DET> bil <NOUN> som <ADP> er " \
                          "<AUX> rød <ADJ> sammen <ADV> med <ADP> Jens-Peter <PROPN> E. <PROPN> Hansen <PROPN>"

        self.assertEqual(sentence.to_tagged_string(), expected_string)

        
if __name__ == '__main__':
    unittest.main()