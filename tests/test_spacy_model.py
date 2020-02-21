import unittest

import spacy

from danlp.download import download_model, DEFAULT_CACHE_DIR, _unzip_process_func
from danlp.models import load_spacy_model


class TestSpacyModel(unittest.TestCase):
    def test_download(self):
        # Download model beforehand
        model_path = download_model('spacy', DEFAULT_CACHE_DIR,
                                    process_func=_unzip_process_func,
                                    verbose=True)

        info = spacy.info(model_path)
        self.assertListEqual(info['pipeline'], ['tagger', 'parser', 'ner'])
        self.assertEqual(info['lang'], 'da')

    def test_predictions(self):

        nlp = load_spacy_model()
        some_text = "jeg hopper på en bil som er rød sammen med Jens-Peter E. Hansen"
        doc = nlp(some_text)

        self.assertEqual(str(doc.ents[0]), 'Jens-Peter E. Hansen')
        self.assertTrue(doc.is_parsed)
        self.assertTrue(doc.is_nered)
        self.assertTrue(doc.is_tagged)

        predicted_pos_tags = [token.tag_ for token in doc]
        asserted_pos_tags = ['PRON', 'VERB', 'ADP', 'DET', 'NOUN', 'ADP',
                             'AUX', 'ADJ', 'ADV', 'ADP', 'PROPN', 'PROPN',
                             'PROPN']

        self.assertListEqual(predicted_pos_tags, asserted_pos_tags)


if __name__ == '__main__':
    unittest.main()
