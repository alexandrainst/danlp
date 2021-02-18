from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ListField, SequenceLabelField
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

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        """
        Takes each predicted cluster and makes it into a labeled `Instance` with only that
        cluster labeled, so we can compute gradients of the loss `on the model's prediction of that
        cluster`.  This lets us run interpretation methods using those gradients.  See superclass
        docstring for more info.
        """
        # Digging into an Instance makes mypy go crazy, because we have all kinds of things where
        # the type has been lost.  So there are lots of `type: ignore`s here...
        predicted_clusters = outputs["clusters"]
        span_field: ListField = instance["spans"]  # type: ignore
        instances = []
        for cluster in predicted_clusters:
            new_instance = instance.duplicate()
            span_labels = [
                0 if (span.span_start, span.span_end) in cluster else -1  # type: ignore
                for span in span_field
            ]  # type: ignore
            new_instance.add_field(
                "span_labels", SequenceLabelField(span_labels, span_field), self._model.vocab
            )
            new_instance["metadata"].metadata["clusters"] = [cluster]  # type: ignore
            instances.append(new_instance)
        if not instances:
            # No predicted clusters; we just give an empty coref prediction.
            new_instance = instance.duplicate()
            span_labels = [-1] * len(span_field)  # type: ignore
            new_instance.add_field(
                "span_labels", SequenceLabelField(span_labels, span_field), self._model.vocab
            )
            new_instance["metadata"].metadata["clusters"] = []  # type: ignore
            instances.append(new_instance)
        return instances

    def _words_list_to_instance(self, sentences: List[List[str]]) -> Instance:
        """
        Create an instance from words lists that represent an already tokenized document
        """
        
        instance = self._dataset_reader.text_to_instance(sentences)
        return instance
