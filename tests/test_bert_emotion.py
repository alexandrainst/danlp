import unittest

from danlp.models import BertEmotion
from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func
from transformers import BertTokenizer, BertForSequenceClassification
import os

class TestBertEmotion(unittest.TestCase):
    def test_download(self):
        # Download model beforehand
        for model in ['bert.emotion', 'bert.noemotion']:
            
            model_path = download_model(model, DEFAULT_CACHE_DIR,
                                        process_func=_unzip_process_func,
                                        verbose=True)
            model_path = os.path.join(model_path,model)
            
            # check if path to model excist
            self.assertTrue(os.path.exists(model_path))
       
            
    def test_predictions(self):
        print('test prdiction')
        model = BertEmotion()
        self.assertTrue(model.predict_if_emotion('bilen er flot')=='Emotional')
        self.assertTrue(model.predict_if_emotion('bilen er rød')=='No emotion')
        self.assertTrue(model.predict('jeg er meget glad idag')=='Glæde/Sindsro')

if __name__ == '__main__':
    unittest.main()
        