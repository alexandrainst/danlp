from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func

import torch
import warnings

from typing import List

class XLMRCoref():
    """
    XLM-Roberta Coreference Resolution Model.

    For predicting which expressions (word or group of words) 
    refer to the same entity in a document. 

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    """
    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):

        from allennlp.models.archival import load_archive
        from allennlp.common.util import import_module_and_submodules
        from allennlp.common.util import prepare_environment
        import_module_and_submodules("danlp.models.allennlp_models")
        from danlp.models.allennlp_models.coref.predictors.coref import CorefPredictor

        # download the model or load the model path
        model_path = download_model('xlmr.coref', cache_dir,
                                      process_func=_unzip_process_func,
                                      verbose=verbose)
                                      
        archive = load_archive(model_path)
        self.config = archive.config
        prepare_environment(self.config)
        self.model = archive.model
        self.dataset_reader = archive.validation_dataset_reader
        self.predictor = CorefPredictor(model=self.model, dataset_reader=self.dataset_reader)
    
    def predict(self, document: List[List[str]]):
        """
        Predict coreferences in a document

        :param List[List[str]] document: segmented and tokenized text
        :return: a dictionary
        :rtype: Dict
        """

        preds = self.predictor.predict_tokenized(document)

        return preds

    def predict_clusters(self, document: List[List[str]]):
        """
        Predict clusters of entities in the document. 
        Each predicted cluster contains a list of references.
        A reference is a tuple (ref text, start id, end id).
        The ids refer to the token ids in the entire document. 
        
        :param List[List[str]] document: segmented and tokenized text
        :return: a list of clusters
        :rtype: List[List[Tuple]]
        """
    
        preds = self.predict(document)
        tokens = [t for d in document for t in d]
        clusters = []

        for idx in preds['clusters']:
            cluster = []
            for ref_idx in idx:
                start_id = ref_idx[0]
                end_id = ref_idx[1]+1
                ref = tokens[start_id:end_id]
                cluster.append((ref, start_id, end_id))
            clusters.append(cluster)

        return clusters



class XlmrNed():
    """
    XLM-Roberta for Named Entity Disambiguation.

    For predicting whether or not a specific entity (QID) is mentioned in a sentence.

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
        #download the model or load the model path
        model_path = download_model('xlmr.ned', cache_dir,
                                     process_func=_unzip_process_func,
                                     verbose=verbose)
        self.classes = ['0', '1']

        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_path, num_labels=len(self.classes))

        self.max_length = self.model.roberta.embeddings.position_embeddings.num_embeddings - 2

    def _classes(self):
        return self.classes
    
    def _get_pred(self, sentence, kg_context):
        input1 = self.tokenizer.encode_plus(sentence, kg_context, add_special_tokens=True, return_tensors='pt',
                                                max_length=self.max_length, truncation='only_second', return_overflowing_tokens=True)
        if 'overflowing_tokens' in input1 and input1['overflowing_tokens'].shape[1]>0:
            warnings.warn('Maximum length for sequence exceeded, truncation may result in unexpected results. Consider running the model on a shorter sequence than {} tokens'.format(self.max_length))
        pred = self.model(input1['input_ids'])[0]

        return pred
    
    def predict(self, sentence: str, kg_context: str):
        """
        Predict whether a QID is mentioned in a sentence or not.

        :param str sentence: raw text
        :param str kg_context: raw text
        :return: 
        :rtype: str
        """

        pred = self._get_pred(sentence, kg_context)
        pred = pred.argmax().item()
        predclass = self.classes[pred]
    
        return predclass
    
    def predict_proba(self, sentence: str, kg_context: str):
        proba=[]
        
        pred=self._get_pred(sentence, kg_context)
        proba.append(torch.nn.functional.softmax(pred[0], dim=0).detach().numpy())

        return proba
        


def load_xlmr_coref_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Loads an XLM-R coreference model.

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    :return: an XLM-R coreference model
    """
    return XLMRCoref(cache_dir, verbose)
        

def load_xlmr_ned_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Loads an XLM-R model for named entity disambiguation.

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    :return: an XLM-R NED model
    """
    return XlmrNed(cache_dir, verbose)

