from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func
from transformers import BertTokenizer, BertForSequenceClassification
import os

class BertEmotion:
    '''
    The class load both a BERT model to classify if emotion or not in the text, and a BERT model to regonizes eight emotions
    '''
    

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        # download the model or load the model path
        path_emotion = download_model('bert.emotion', cache_dir,
                                       process_func=_unzip_process_func,
                                       verbose=verbose)
        path_emotion = os.path.join(path_emotion,'bert.emotion')
        path_reject = download_model('bert.noemotion', cache_dir,
                                       process_func=_unzip_process_func,
                                       verbose=verbose)
        path_reject = os.path.join(path_reject,'bert.noemotion')
        # load the models
        self.tokenizer_rejct = BertTokenizer.from_pretrained(path_reject)
        self.model_reject = BertForSequenceClassification.from_pretrained(path_reject)
        
        self.tokenizer = BertTokenizer.from_pretrained(path_emotion)
        self.model = BertForSequenceClassification.from_pretrained(path_emotion)
        
        # load the class names mapping
        self.catagories={5:'Foragt/Modvilje', 2:'Forventning/Interrese',
                   0:'Glæde/Sindsro', 3:'Overasket/Målløs', 1:'Tillid/Accept',
                   4:'Vrede/Irritation',  6:'Sorg/trist',7:'Frygt/Bekymret'}
    
    def predict_if_emotion(self, sentence):
        labels = {1: 'No emotion', 0: 'Emotional'}
        input1 = self.tokenizer_rejct.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
        pred = self.model_reject(input1['input_ids'], token_type_ids=input1['token_type_ids'])[0].argmax().item()
        return labels[pred]
        
    
    def predict(self, sentence: str, no_emotion=False):
        
        def predict_emotion():
            input1 = self.tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
            pred = self.model(input1['input_ids'], token_type_ids=input1['token_type_ids'])[0].argmax().item() 
            return self.catagories[pred]
        
        if no_emotion:
            return predict_emotion()
        else:
            reject=self.predict_if_emotion(sentence)
            if reject=='No emotion':
                return reject
            else:
                return predict_emotion()

            
def load_bert_emotion_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    # wrapper function to ensure that all models in danlp is loaded in a similar way
    return  BertEmotion(cache_dir, verbose)