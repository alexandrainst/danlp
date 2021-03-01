"""
This a modified version of the test_spacy_model.py adapted for DaCy
"""
import os
import unittest
import spacy
from danlp.models import load_spacy_chunking_model


class TestSpacyModel(unittest.TestCase):

    def test_predictions(self):
        # nlp = load_spacy_model()
        nlp = spacy.load(DACY_PATH)
        some_text = "Jeg gik en tur med Lars Bo Jensen i går"
        doc = nlp(some_text)
        self.assertTrue(doc.is_parsed)
        self.assertTrue(doc.is_nered)
        self.assertTrue(doc.is_tagged)

        # updating the following would break compatibility with previous models test thus it is not done
        # (these would have to be changed regardless to support the new structure of SpaCy v3)

        # chunker = load_spacy_chunking_model(spacy_model=nlp)
        # chunks_from_text = chunker.predict(some_text)
        # chunks_from_tokens = chunker.predict([t.text for t in doc])
        # self.assertEqual(chunks_from_text, chunks_from_tokens)
        # self.assertEqual(len(chunks_from_text), len(doc))


if __name__ == '__main__':
    DACY_PATH = os.path.expanduser(
        "~/desktop/package/da_dacy_large_tft-0.0.0/da_dacy_large_tft/da_dacy_large_tft-0.0.0")
    # path = os.path.expanduser('~/desktop/package')
    # os.listdir(path)
    unittest.main()
