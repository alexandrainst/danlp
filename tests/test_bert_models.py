import unittest

from danlp.models import load_bert_emotion_model, load_bert_tone_model, load_bert_base_model, BertNer
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
        model = load_bert_emotion_model()
        self.assertTrue(model.predict_if_emotion('bilen er flot')=='Emotional')
        self.assertTrue(model.predict_if_emotion('bilen er rød')=='No emotion')
        self.assertTrue(model.predict('jeg er meget glad idag')=='Glæde/Sindsro')
        self.assertTrue(len(model.predict_proba('jeg er meget glad idag')[0])==8)
        self.assertTrue(len(model.predict_proba('jeg er meget glad idag', no_emotion=True)[1])==2)
        self.assertEqual(model._classes()[0], ['Glæde/Sindsro','Tillid/Accept','Forventning/Interrese','Overasket/Målløs','Vrede/Irritation','Foragt/Modvilje','Sorg/trist','Frygt/Bekymret'])
        self.assertEqual(model._classes()[1], ['No emotion', 'Emotional'])

class TestBertTone(unittest.TestCase):
    def test_download(self):
        # Download model beforehand
        for model in ['bert.subjective', 'bert.polarity']:
            version= {'bert.subjective': 'bert.sub.v0.0.1', 'bert.polarity': 'bert.pol.v0.0.1'}
            model_path = download_model(model, DEFAULT_CACHE_DIR,
                                        process_func=_unzip_process_func,
                                        verbose=True)
            model_path = os.path.join(model_path,version[model])
            
            # check if path to model excist
            self.assertTrue(os.path.exists(model_path))

    def test_predictions(self):
        model = load_bert_tone_model()
        self.assertEqual(model.predict('han er 12 år', polarity=False),{'analytic': 'objective', 'polarity': None})
        self.assertEqual(model.predict('han gør det godt', analytic=False),{'analytic': None, 'polarity': 'positive'})
        self.assertEqual(model.predict('Det er super dårligt'),{'analytic': 'subjective', 'polarity': 'negative'})
        self.assertEqual(model._classes()[0],  ['positive', 'neutral', 'negative'])
        self.assertTrue(len(model.predict_proba('jeg er meget glad idag', polarity=False)[0])==2)

class TestBertTone(unittest.TestCase):
    def test_download(self):
        # Download model beforehand
            model = 'bert.botxo.pytorch'
            model_path = download_model(model, DEFAULT_CACHE_DIR,
                                        process_func=_unzip_process_func,
                                        verbose=True)
            
            # check if path to model excist
            self.assertTrue(os.path.exists(model_path))
            
    def test_embedding(self):
        model = load_bert_base_model()
        vecs_embedding, sentence_embedding, tokenized_text =model.embed_text('Han sælger frugt')
        self.assertEqual(len(vecs_embedding),3)
        self.assertEqual(vecs_embedding[0].shape[0], 3072)
        self.assertEqual(sentence_embedding.shape[0], 768)

class TestBertNer(unittest.TestCase):
    def test_bert_tagger(self):
        bert = BertNer()
        tokens, prediction = bert.predict("Jeg var ude og gå i København")

        self.assertEqual(len(tokens), len(prediction))
        self.assertEqual(prediction, ['O', 'O', 'O', 'O', 'O', 'O', 'B-LOC'])

        tokenized_string = ["Begge", "de", "to", "bankers", "økonomiske", "\"",
                            "engagement", "\"", "i", "Brøndby", "er", "for",
                            "nærværende", "så", "eksklusivt", ",", "at", "de",
                            "-", "qua", "konkursbegæringer", "-", "begge",
                            "den", "dag", "i", "går", "i", "praksis", "kunne",
                            "have", "sparket", "Brøndby", "langt", "ud", "af",
                            "dansk", "topfodbold", "."]

        tokens, prediction = bert.predict(tokenized_string)

        self.assertEqual(len(tokenized_string), len(prediction))


if __name__ == '__main__':
    unittest.main()
