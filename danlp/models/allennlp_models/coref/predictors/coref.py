from typing import List, Dict

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register("coreference_resolution")
class CorefPredictor(Predictor):
    """
    Predictor for the [`CoreferenceResolver`](../models/coreference_resolution/coref.md) model.

    Registered as a `Predictor` with name "coreference_resolution".
    """

    def __init__(
        self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict_tokenized(self, tokenized_sentences: List[List[str]]) -> JsonDict:
        """
        Predict the coreference clusters in the given document.

        # Parameters

        tokenized_sentences : `List[str]`
            A list of words representation of a tokenized document.

        # Returns

        A dictionary representation of the predicted coreference clusters.
        """
        instance = self._words_list_to_instance(tokenized_sentences)
        return self.predict_instance(instance)

    def _words_list_to_instance(self, sentences: List[List[str]]) -> Instance:
        """
        Create an instance from words lists that represent an already tokenized document
        """
        
        instance = self._dataset_reader.text_to_instance(sentences)
        return instance
