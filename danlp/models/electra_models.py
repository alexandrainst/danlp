import os
from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func
import torch
import warnings

class ElectraOffensive():
    """
    Electra Offensive Model.

    For detecting whether a comment is offensive or not. 

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    """
    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        from transformers import ElectraTokenizer, ElectraForSequenceClassification
        # download the model or load the model path
        model_path = download_model('electra.offensive', cache_dir, process_func=_unzip_process_func,verbose=verbose)
        
        self.classes = ['NOT', 'OFF']

        self.tokenizer = ElectraTokenizer.from_pretrained(model_path)
        self.model = ElectraForSequenceClassification.from_pretrained(model_path, num_labels=len(self.classes))
        
        self.max_length = self.model.electra.embeddings.position_embeddings.num_embeddings


    def _classes(self):
        return self.classes
    

    def _get_pred(self, sentence):
        input1 = self.tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt',
                                                max_length=self.max_length, truncation=True, return_overflowing_tokens=True)
        if 'overflowing_tokens' in input1 and input1['overflowing_tokens'].shape[1]>0:
            warnings.warn('Maximum length for sequence exceeded, truncation may result in unexpected results. Consider running the model on a shorter sequence than {} tokens'.format(self.max_length))
        pred = self.model(input1['input_ids'])[0]

        return pred
    
    def predict(self, sentence: str):
        """
        Predict whether a sentence is offensive or not

        :param str sentence: raw text
        :return: a class representing whether the sentence is offensive or not (`OFF`/`NOT`)
        :rtype: str
        """
        
        pred = self._get_pred(sentence)
        pred = pred.argmax().item()
        predclass = self.classes[pred]

        return predclass

    def predict_proba(self, sentence: str):
        """
        For a given sentence, 
        return its probabilities of belonging to each class, 
        i.e. `OFF` or `NOT`
        """

        pred = self._get_pred(sentence)
        proba = torch.nn.functional.softmax(pred[0], dim=0).detach().numpy()

        return proba


def load_electra_offensive_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Loads an Electra Offensive model.

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    :return: an Electra Offensive model
    """
    return ElectraOffensive(cache_dir, verbose)