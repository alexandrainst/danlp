import unittest

from flair.data import Sentence

from danlp.download import download_model, DEFAULT_CACHE_DIR, _unzip_process_func
from danlp.models import load_ner_tagger_with_flair


class TestNerTaggers(unittest.TestCase):
    def test_flair_tagger(self):
        # Download model beforehand
        download_model('flair.ner', DEFAULT_CACHE_DIR, process_func=_unzip_process_func, verbose=True)
        print("Downloaded the flair model")

        # Load the NER tagger using the DaNLP wrapper
        flair_model = load_ner_tagger_with_flair()

        # Using the flair POS tagger
        sentence = Sentence('jeg hopper på en bil som er rød sammen med Jens-Peter E. Hansen')
        flair_model.predict(sentence)

        expected_string = "jeg hopper på en bil som er rød sammen med Jens-Peter <B-PER> E. <I-PER> Hansen <I-PER>"

        self.assertEqual(sentence.to_tagged_string(), expected_string)


if __name__ == '__main__':
    unittest.main()