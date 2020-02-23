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
        some_text = "Jeg gik en tur med Lars"
        doc = nlp(some_text)
        self.assertTrue(doc.is_parsed)
        self.assertTrue(doc.is_nered)
        self.assertTrue(doc.is_tagged)


if __name__ == '__main__':
    unittest.main()
