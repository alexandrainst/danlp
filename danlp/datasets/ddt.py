import os

import pyconll

from danlp.download import DEFAULT_CACHE_DIR, download_dataset, _unzip_process_func, DATASETS


def _any_part_exist(parts: list):
    for part in parts:
        if part is not None:
            return True
    return False


class DDT:
    """
    The DDT dataset has been annotated with NER tags in the IOB2 format.
    The dataset is downloaded in CoNLL-U format, but with this class
    it can be converted to spaCy format or a simple NER format
    similar to the CoNLL 2003 NER format.

    """
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.dataset_name = 'ddt'
        self.file_extension = DATASETS[self.dataset_name]['file_extension']
        self.dataset_dir = download_dataset('ddt', process_func=_unzip_process_func, cache_dir=cache_dir)

    def load_as_conllu(self, predefined_splits: bool = False):
        """

        :param predefined_splits:
        :return A single pyconll.Conll
                or a tuple of (train, dev, test) pyconll.Conll
                depending on predefined_split
        """

        parts = [None, None, None]  # Placeholder list to put predefined parts of dataset [train, dev, test]
        for i, part in enumerate(['train', 'dev', 'test']):
            file_name = "{}.{}{}".format(self.dataset_name, part, self.file_extension)
            file_path = os.path.join(self.dataset_dir, file_name)

            parts[i] = pyconll.load_from_file(file_path)

        # if predefined_splits: then we should return three files
        if predefined_splits:
            return parts

        # Merge the splits to one single dataset
        parts[0].extend(parts[1])
        parts[0].extend(parts[2])

        return parts[0]

    def load_as_simple_ner(self, predefined_splits: bool = False):
        conllu_parts = self.load_as_conllu(predefined_splits)

        if not predefined_splits:
            conllu_parts = [conllu_parts]

        parts = []
        for conllu_part in conllu_parts:
            part_sentences = []
            part_entities = []

            for sent in conllu_part:
                part_sentences.append([token.form for token in sent._tokens])
                part_entities.append([token.misc['name'].pop() for token in sent._tokens])

            parts.append([part_sentences, part_entities])

        if predefined_splits:
            return parts
        return parts[0]


    def load_with_flair(self, predefined_splits: bool = False):
        """
        This function is inspired by the "Reading Your Own Sequence Labeling Dataset" from Flairs tutorial
        on reading corpora:

        https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_6_CORPUS.md

        TODO: Make a pull request to flair similar to this:
        https://github.com/zalandoresearch/flair/issues/383

        :param predefined_splits:
        :return: ColumnCorpus
        """

        from flair.data import Corpus
        from flair.datasets import ColumnCorpus

        columns = {1: 'text', 3: 'pos', 9: 'ner'}

        # init a corpus using column format, data folder and the names of the train, dev and test files
        corpus: Corpus = ColumnCorpus(self.dataset_dir, columns, comment_symbol='#',
                                      train_file='{}.{}{}'.format(self.dataset_name, 'train', self.file_extension),
                                      test_file='{}.{}{}'.format(self.dataset_name, 'test', self.file_extension),
                                      dev_file='{}.{}{}'.format(self.dataset_name, 'dev', self.file_extension))

        # Remove the `name=` from `name=B-PER` to only use the `B-PER` tag
        parts = ['train', 'dev', 'test']
        for part in parts:
            dataset = corpus.__getattribute__(part)

            for sentence in dataset.sentences:
                for token in sentence.tokens:
                    if 'ner' in token.tags:
                        token.tags['ner'].value = token.tags['ner'].value.split("=")[1].replace("|SpaceAfter", "")

        return corpus

    def load_with_spacy(self):
        """
        Converts the conllu files to json in the spaCy format.

        Not using jsonl because of:
        https://github.com/explosion/spaCy/issues/3523
        :return:
        """
        import srsly
        from spacy.cli.converters import conllu2json
        from spacy.gold import GoldCorpus
        from spacy.gold import Path

        for part in ['train', 'dev', 'test']:
            conll_path = os.path.join(self.dataset_dir, '{}.{}{}'.format(self.dataset_name, part, self.file_extension))
            json_path = os.path.join(self.dataset_dir, "ddt.{}.json".format(part))

            if not os.path.isfile(json_path):  # Convert the conllu files to json
                with open(conll_path, 'r') as file:
                    file_as_string = file.read()
                    file_as_string = file_as_string.replace("name=", "").replace("|SpaceAfter", "")
                    file_as_json = conllu2json(file_as_string)

                    srsly.write_json(json_path, file_as_json)

        train_json_path = os.path.join(self.dataset_dir, "ddt.train.json")
        dev_json_path = os.path.join(self.dataset_dir, "ddt.dev.json")

        assert os.path.isfile(train_json_path)
        assert os.path.isfile(dev_json_path)

        return GoldCorpus(Path(train_json_path), Path(dev_json_path))
