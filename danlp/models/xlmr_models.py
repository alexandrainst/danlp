from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func

from allennlp.models.archival import load_archive
from allennlp.common.util import import_module_and_submodules
from allennlp.common.util import prepare_environment

import_module_and_submodules("danlp.models.allennlp_models")
from danlp.models.allennlp_models.coref.predictors.coref import CorefPredictor

import os
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


def load_xlmr_coref_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Loads an XLM-R coreference model.

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    :return: an XLM-R coreference model
    """
    return XLMRCoref(cache_dir, verbose)