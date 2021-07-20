
import unittest

from danlp.models import load_xlmr_coref_model
from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func

from allennlp.data.data_loaders import SimpleDataLoader 
from allennlp.training.util import evaluate

import os


class TestXLMRCoref(unittest.TestCase):
    def test_download(self):
        model_name = 'xlmr.coref'
        # Download model beforehand
        model_path = download_model(model_name, DEFAULT_CACHE_DIR,
                                      process_func=_unzip_process_func,
                                      verbose=True)
        # check if path to model exists
        self.assertTrue(os.path.exists(model_path))

    def test_model(self):
        xlmr_model = load_xlmr_coref_model()

        doc = [["Lotte", "arbejder", "med", "Mads", "."], ["Hun", "er", "tandlæge", "."]]

        # prediction
        preds = xlmr_model.predict(doc)
        clusters = xlmr_model.predict_clusters(doc)

        self.assertEqual(preds['top_spans'], [[0, 0], [1, 3], [5, 5]])
        self.assertEqual(preds['antecedent_indices'], [[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        self.assertEqual(preds['predicted_antecedents'], [-1, -1, 0])
        self.assertEqual(preds['clusters'], [[[0, 0], [5, 5]]])
        self.assertEqual(clusters, [[(['Lotte'], 0, 1), (['Hun'], 5, 6)]])

        # evaluation
        data_loader_params = xlmr_model.config.pop("data_loader")

        from collections import OrderedDict
        sentences = [
                [
                    OrderedDict([('id', 1), ('form', 'Lotte'), ('lemma', 'Lotte'), ('upos', 'PROPN'), ('coref_rel', '(1086)'), ('doc_id', '1'), ('qid', '-')]),
                    OrderedDict([('id', 2), ('form', 'arbejder'), ('lemma', 'arbejde'), ('upos', 'VERB'), ('coref_rel', '-'), ('doc_id', '1'), ('qid', '-')]),
                    OrderedDict([('id', 3), ('form', 'med'), ('lemma', 'med'), ('upos', 'ADV'), ('coref_rel', '-'), ('doc_id', '1'), ('qid', '-')]),
                    OrderedDict([('id', 4), ('form', 'Mads'), ('lemma', 'Mads'), ('upos', 'PROPN'), ('coref_rel', '(902)'), ('doc_id', '1'), ('qid', '-')]),
                    OrderedDict([('id', 5), ('form', '.'), ('lemma', '.'), ('upos', 'PUNCT'), ('coref_rel', '-'), ('doc_id', '1'), ('qid', '-')])
                ],
                [
                    OrderedDict([('id', 1), ('form', 'Hun'), ('lemma', 'hun'), ('upos', 'PRON'), ('coref_rel', '(1086)'), ('doc_id', '1'), ('qid', '-')]),
                    OrderedDict([('id', 2), ('form', 'er'), ('lemma', 'vær'), ('upos', 'VERB'), ('coref_rel', '-'), ('doc_id', '1'), ('qid', '-')]),
                    OrderedDict([('id', 3), ('form', 'tandlæge'), ('lemma', 'tandlæge'), ('upos', 'NOUN'), ('coref_rel', '-'), ('doc_id', '1'), ('qid', '-')]),
                    OrderedDict([('id', 5), ('form', '.'), ('lemma', '.'), ('upos', 'PUNCT'), ('coref_rel', '-'), ('doc_id', '1'), ('qid', '-')])
                ]
            ]
        
        instances = xlmr_model.dataset_reader.load_dataset(sentences)
        data_loader = SimpleDataLoader(instances, 1)
        data_loader.index_with(xlmr_model.model.vocab)

        metrics = evaluate(
            xlmr_model.model,
            data_loader
        )
        
        self.assertEqual(metrics['coref_precision'], 1.0)

