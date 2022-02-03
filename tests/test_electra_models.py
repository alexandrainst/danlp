import unittest

from danlp.models import load_electra_offensive_model
from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func
import os


class TestElectraOffensive(unittest.TestCase):

    def test_download(self):
        model_name = 'electra.offensive'
        # Download model beforehand
        model_path = download_model(model_name, DEFAULT_CACHE_DIR,
                                      process_func=_unzip_process_func,
                                      verbose=True)
        # check if path to model exists
        self.assertTrue(os.path.exists(model_path))

    def test_predictions(self):
        model = load_electra_offensive_model()
        sentence = "Han ejer ikke respekt for nogen eller noget... han er megaloman og psykopat"
        self.assertEqual(model.predict(sentence), "OFF")
        self.assertEqual(model._classes(), ['NOT', 'OFF'])
        self.assertTrue(len(model.predict_proba(sentence))==2)



if __name__ == '__main__':
    unittest.main()
