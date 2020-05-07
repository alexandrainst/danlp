import os
from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func


class BertNer:
    """
    Bert NER model
    """
    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        # download the model or load the model path
        weights_path = download_model('bert.ner', cache_dir,
                                      process_func=_unzip_process_func,
                                      verbose=verbose)

        self.label_list = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG",
                           "I-ORG", "B-LOC", "I-LOC"]

        self.model = AutoModelForTokenClassification.from_pretrained(weights_path)
        self.tokenizer = AutoTokenizer.from_pretrained(weights_path)

    def predict(self, text):
        import torch
        # Bit of a hack to get the tokens with the special tokens
        tokens = self.tokenizer.tokenize(
            self.tokenizer.decode(self.tokenizer.encode(text)))
        inputs = self.tokenizer.encode(text, return_tensors="pt")

        outputs = self.model(inputs)[0]
        predictions = torch.argmax(outputs, dim=2)

        return [self.label_list[label] for label in predictions[0].tolist()]


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
        self.catagories = {5: 'Foragt/Modvilje', 2: 'Forventning/Interrese',
                           0: 'Glæde/Sindsro', 3: 'Overasket/Målløs',
                           1: 'Tillid/Accept',
                           4: 'Vrede/Irritation', 6: 'Sorg/trist',
                           7: 'Frygt/Bekymret'}
    
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
    """
    Wrapper function to ensure that all models in danlp are
    loaded in a similar way
    :param cache_dir:
    :param verbose:
    :return:
    """

    return BertEmotion(cache_dir, verbose)
