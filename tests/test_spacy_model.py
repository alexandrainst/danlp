import unittest

import spacy
import operator
from danlp.download import download_model, DEFAULT_CACHE_DIR, _unzip_process_func
from danlp.models import load_spacy_model
import os

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
        
class TestSpacySentimentModel(unittest.TestCase):
    def test_download(self):
        # Download model beforehand
        model_path = download_model('spacy.sentiment', DEFAULT_CACHE_DIR,
                                    process_func=_unzip_process_func,
                                    verbose=True)

        info = spacy.info(os.path.join(model_path, 'spacy.sentiment'))
        self.assertListEqual(info['pipeline'], ['tagger', 'parser', 'ner', 'textcat'])
        self.assertEqual(info['lang'], 'da')
        

    def test_predictions(self):

        nlp = load_spacy_model(textcat='sentiment')
        some_text = "Vi er glade for spacy!"
        doc = nlp(some_text)
        self.assertTrue(doc.is_parsed)
        self.assertTrue(doc.is_nered)
        self.assertTrue(doc.is_tagged)
        self.assertEqual(max(doc.cats.items(), key=operator.itemgetter(1))[0],'positiv')

if __name__ == '__main__':
    unittest.main()
