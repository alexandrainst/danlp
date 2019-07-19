import unittest

from flair.data import Sentence

from danlp.download import download_model, DEFAULT_CACHE_DIR
from danlp.models.pos_taggers import load_pos_tagger_with_flair


class TestPosTaggers(unittest.TestCase):
    def test_flair_tagger(self):
        # Download model beforehand
        download_model('flair.pos', DEFAULT_CACHE_DIR, verbose=True)
        print("Downloaded the flair model")

        # Load the POS tagger using the DaNLP wrapper
        flair_model = load_pos_tagger_with_flair()

        # Using the flair POS tagger
        sentence = Sentence('jeg hopper på en bil som er rød sammen med Jens-Peter E. Hansen')
        flair_model.predict(sentence)

        expected_string = "jeg <PRON> hopper <VERB> på <ADP> en <DET> bil <NOUN> som <ADP> er " \
                          "<AUX> rød <ADJ> sammen <ADV> med <ADP> Jens-Peter <PROPN> E. <PROPN> Hansen <PROPN>"

        self.assertEqual(sentence.to_tagged_string(), expected_string)


if __name__ == '__main__':
    unittest.main()
