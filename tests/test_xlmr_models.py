import unittest

from danlp.models import load_xlmr_coref_model
from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func
import os


class TestXLMRCoref(unittest.TestCase):
    def test_download(self):
        model_name = 'xlmr.coref'
        # Download model beforehand
        model_path = download_model(model_name, DEFAULT_CACHE_DIR,
                                      process_func=_unzip_process_func,
                                      verbose=True)
        # check if path to model exists
        self.assertTrue(os.path.exists(model_path))

    def test_predictions(self):
        model = load_xlmr_coref_model()

        doc = [["Lotte", "arbejder", "med", "Mads", "."], ["Hun", "er", "tandlæge", "."]]
        preds = model.predict(doc)

        self.assertEqual(preds['top_spans'], [[0, 0], [1, 3], [5, 5]])
        self.assertEqual(preds['antecedent_indices'], [[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        self.assertEqual(preds['predicted_antecedents'], [-1, -1, 0])
        self.assertEqual(preds['clusters'], [[[0, 0], [5, 5]]])
        
