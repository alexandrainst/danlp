import os
import conllu
import json
from danlp.download import DEFAULT_CACHE_DIR, download_dataset, _unzip_process_func, DATASETS


class Dacoref:
    """
    This Danish coreference annotation contains parts of the Copenhagen Dependency Treebank. 
    It was originally annotated as part of the Copenhagen Dependency Treebank (CDT) project but never finished. 
    This resource extends the annotation by using different mapping techniques and by augmenting with Qcodes from Wiktionary. 
    Read more about it in the danlp docs.

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity

    """

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.dataset_name = 'dacoref'
        self.dataset_dir = download_dataset(self.dataset_name, process_func=_unzip_process_func, cache_dir=cache_dir)

    def load_as_conllu(self, predefined_splits: bool = False):
        """
        :param bool predefined_splits: Boolean
        :return: A single parsed conllu list
                or a list of train, dev, test split parsed conllu list
                depending on predefined_split
        """
        with open('{}/CDT_coref.conllu'.format(self.dataset_dir)) as f:
            conlist = conllu.parse(f.read(), fields=["id", "form", "lemma", "upos", 'xpos', 'feats', 'head', 'deprel','deps', 'misc', 'coref_id', 'coref_rel', 'doc_id', 'qid'])

        if predefined_splits==False:
            return conlist

        parts = [None, None, None] 
        sent_parts = [[], [], []]
        for i, part in enumerate(['train', 'dev', 'test']):
            with open('{}/CDT_{}_ids.json'.format(self.dataset_dir,part)) as f:
                parts[i] = json.load(f)
            for sentence in conlist:
                if sentence[0]["doc_id"] in parts[i]:
                    sent_parts[i].append(sentence)


        return sent_parts