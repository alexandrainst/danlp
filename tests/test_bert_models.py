import unittest

from danlp.models import load_bert_emotion_model, load_bert_tone_model, load_bert_base_model, load_bert_ner_model, load_bert_nextsent_model, load_bert_offensive_model
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
        self.assertTrue(model.predict('bilen er rød')=='No emotion')
        self.assertTrue(model.predict('jeg er meget glad idag')=='Glæde/Sindsro')
        self.assertTrue(model.predict('jeg er meget glad idag', no_emotion=True)=='Glæde/Sindsro')
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

class TestBertBase(unittest.TestCase):
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

class TestBertNextSent(unittest.TestCase):  
    def test_next_sent(self):
        model = load_bert_nextsent_model()
        
        sent_A= "Uranus er den syvende planet fra Solen i Solsystemet og var den første planet der blev opdaget i historisk tid."
        sent_B1 =" William Herschel opdagede d. 13. marts 1781 en tåget klat, som han først troede var en fjern komet." 
        sent_B2= "Yderligere er magnetfeltets akse 59° forskudt for rotationsaksen og skærer ikke centrum."
        
        self.assertTrue(model.predict_if_next_sent(sent_A, sent_B1) >0.75)
        self.assertTrue(model.predict_if_next_sent(sent_A, sent_B2) <0.75)
        
class TestBertNer(unittest.TestCase):
    def test_bert_tagger(self):
        bert = load_bert_ner_model()
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

        # test with non IBOformat:
        tekst_tokenized = ['Han', 'hedder', 'Anders', 'And', 'Andersen', 'og', 'bor', 'i', 'Århus', 'C']
        dict_pred = bert.predict(tekst_tokenized, IOBformat=False)
        self.assertEqual(dict_pred['entities'][0]['text'], 'Anders And Andersen')
        self.assertEqual(dict_pred['entities'][1]['type'], 'LOC')

class TestBertOffensive(unittest.TestCase):

    def test_download(self):
        model_name = 'bert.offensive'
        # Download model beforehand
        model_path = download_model(model_name, DEFAULT_CACHE_DIR,
                                      process_func=_unzip_process_func,
                                      verbose=True)
        # check if path to model exists
        self.assertTrue(os.path.exists(model_path))


    def test_predictions(self):
        model = load_bert_offensive_model()
        sentence = "Han ejer ikke respekt for nogen eller noget... han er megaloman og psykopat"
        self.assertEqual(model.predict(sentence), "OFF")
        self.assertEqual(model._classes(), ['NOT', 'OFF'])
        self.assertTrue(len(model.predict_proba(sentence))==2)
        
if __name__ == '__main__':
    unittest.main()
