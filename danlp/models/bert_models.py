import os, re
from typing import Union, List
import tensorflow as tf
from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func


class BertNer:
    """
    Bert NER model
    """
    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        from transformers import AutoModelForTokenClassification
        from transformers import AutoTokenizer

        # download the model or load the model path
        weights_path = download_model('bert.ner', cache_dir,
                                      process_func=_unzip_process_func,
                                      verbose=verbose)

        self.label_list = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG",
                           "I-ORG", "B-LOC", "I-LOC"]

        self.model = AutoModelForTokenClassification.from_pretrained(weights_path)
        self.tokenizer = AutoTokenizer.from_pretrained(weights_path)

    def predict(self, text: Union[str, List[str]]):
        """
        Predict NER labels from raw text or tokenized text. If the text is
        a raw string this method will return the string tokenized with
        BERTs subword tokens.

        E.g. "varme vafler" will become ["varme", "va", "##fler"]

        :param text: Can either be a raw text or a list of tokens
        :return: The tokenized text and the predicted labels
        """
        import torch

        if isinstance(text, str):
            # Bit of a hack to get the tokens with the special tokens
            tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(text)))
            inputs = self.tokenizer.encode(text, return_tensors="pt")
            outputs = self.model(inputs)[0]
            predictions = torch.argmax(outputs, dim=2)
            predictions = [self.label_list[label] for label in
                           predictions[0].tolist()]

            return tokens[1:-1], predictions[1:-1]  # Remove special tokens

        if isinstance(text, list):
            # Tokenize each word into the subword tokenization that BERT
            # uses. E.g. this tokenized text:
            #     tokens: ['Varme', 'vafler', 'og', 'friske', 'jordbær']
            # will get the following subword tokens and token_mask
            #     subwords: ['varme', 'va', '##fler', 'og', 'friske', 'jordbær']
            #     tokens_mask: [1, 1, 1, 0, 1, 1, 1, 1]
            tokens = []
            tokens_mask = [1]
            for word in text:
                word_tokens = self.tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                tokens_mask.extend([1]+[0]*(len(word_tokens)-1))
            tokens_mask.extend([1])

            inputs = self.tokenizer.encode(tokens, return_tensors="pt")
            assert inputs.shape[1] == len(tokens_mask)

            outputs = self.model(inputs)[0]

            predictions = torch.argmax(outputs, dim=2)

            # Mask the predictions so we only get the labels for the
            # pre-tokenized text
            predictions = [prediction for prediction, mask in zip(predictions[0].tolist(), tokens_mask) if mask]
            predictions = predictions[1:-1]  # Remove special tokens
            assert len(predictions) == len(text)

            # Map prediction ids to labels
            predictions = [self.label_list[label] for label in predictions]

            return text, predictions

class BertEmotion:
    """
    The class load both a BERT model to classify if emotion or not in the text,
    and a BERT model to regonizes eight emotions
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        from transformers import BertTokenizer, BertForSequenceClassification
        
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
        self.catagories = { 0: 'Glæde/Sindsro',1: 'Tillid/Accept', 2: 'Forventning/Interrese',
              3: 'Overasket/Målløs',4: 'Vrede/Irritation', 5: 'Foragt/Modvilje', 6: 'Sorg/trist',7: 'Frygt/Bekymret'}

        self.labels_no = {1: 'No emotion', 0: 'Emotional'}
    
    def _classes(self):
        return list(self.catagories.values()), list(self.labels_no.values())
    
    def predict_if_emotion(self, sentence):
        
        input1 = self.tokenizer_rejct.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
        pred = self.model_reject(input1['input_ids'], token_type_ids=input1['token_type_ids'])[0].argmax().item()
        return self.labels_no[pred]
    
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
            
    def predict_proba(self, sentence: str, no_emotion=False):
        proba=[]
            
        # which emotion        
        input1 = self.tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
        pred = (self.model(input1['input_ids'], token_type_ids=input1['token_type_ids'])[0].detach().numpy())
        proba.append(tf.nn.softmax(pred)[0].numpy())
        
                
        # emotion or no emotion
        if no_emotion:
            input1 = self.tokenizer_rejct.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
            pred = (self.model_reject(input1['input_ids'], token_type_ids=input1['token_type_ids'])[0].detach().numpy())
            proba.append(tf.nn.softmax(pred)[0].numpy())

        return proba 

class BertTone:
    '''
    The class load both a BERT model to classify boteh the tone of [subjective or objective] and the tone og [positive, neutral , negativ]
    returns: [label_subjective, label_polarity]
    '''
   

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):

        from transformers import BertTokenizer, BertForSequenceClassification
        # download the model or load the model path
        path_sub = download_model('bert.subjective', cache_dir, process_func=_unzip_process_func,verbose=verbose)
        path_sub = os.path.join(path_sub,'bert.sub.v0.0.1')
        path_pol = download_model('bert.polarity', cache_dir, process_func=_unzip_process_func,verbose=verbose)
        path_pol = os.path.join(path_pol,'bert.pol.v0.0.1')
        
        self.tokenizer_sub = BertTokenizer.from_pretrained(path_sub)
        self.model_sub = BertForSequenceClassification.from_pretrained(path_sub)
        self.tokenizer_pol = BertTokenizer.from_pretrained(path_pol)
        self.model_pol = BertForSequenceClassification.from_pretrained(path_pol)
        
        self.classes_pol= ['positive', 'neutral', 'negative']
        self.classes_sub= ['objective','subjective'] 
            
    def _clean(self, sentence):
        sentence=re.sub(r'http[^\s]+', '', sentence)
        sentence = re.sub(r'\n', '',sentence)
        sentence = re.sub(r'\t', '',sentence)
        sentence = re.sub(r'@', '',sentence)
        sentence = re.sub(r'#', '',sentence)
        return sentence

    def _classes(self):
        return self.classes_pol, self.classes_sub
    
    def predict(self, sentence: str, polarity: bool = True, analytic: bool = True):

        
        sentence = self._clean(str(sentence))
        predDict = {'analytic': None, 'polarity': None }
        
        # predict subjective
        if analytic:
            input1 = self.tokenizer_sub.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
            pred = self.model_sub(input1['input_ids'], token_type_ids=input1['token_type_ids'])[0].argmax().item()
            predDict['analytic']=self.classes_sub[pred]
        
        # predict polarity
        if polarity:
            input1 = self.tokenizer_pol.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
            pred = self.model_pol(input1['input_ids'], token_type_ids=input1['token_type_ids'])[0].argmax().item() 
            predDict['polarity']=self.classes_pol[pred]
        
        return predDict
    
    def predict_proba(self, sentence: str, polarity: bool = True, analytic: bool = True):
        proba=[]
        sentence = self._clean(str(sentence))
        
        # predict polarity
        if polarity:
            input1 = self.tokenizer_pol.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
            pred = (self.model_pol(input1['input_ids'], token_type_ids=input1['token_type_ids'])[0].detach().numpy())
            proba.append(tf.nn.softmax(pred)[0].numpy())
            
        # predict subjective
        if analytic:          
            input1 = self.tokenizer_sub.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
            pred = (self.model_sub(input1['input_ids'], token_type_ids=input1['token_type_ids'])[0].detach().numpy())
            proba.append(tf.nn.softmax(pred)[0].numpy())

        return proba    
            
def load_bert_emotion_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Wrapper function to ensure that all models in danlp are
    loaded in a similar way
    :param cache_dir:
    :param verbose:
    :return:
    """

    return BertEmotion(cache_dir, verbose)

def load_bert_tone_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Wrapper function to ensure that all models in danlp are
    loaded in a similar way
    :param cache_dir:
    :param verbose:
    :return:
    """

    return BertTone(cache_dir, verbose)


def load_bert_ner_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Wrapper function to ensure that all models in danlp are
    loaded in a similar way
    :param cache_dir:
    :param verbose:
    :return:
    """
    return BertNer(cache_dir, verbose)
