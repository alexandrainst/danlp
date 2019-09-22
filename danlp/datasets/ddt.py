import os

from danlp.download import DEFAULT_CACHE_DIR, download_dataset, _unzip_process_func, DATASETS


def _any_part_exist(parts: list):
    for part in parts:
        if part is not None:
            return True
    return False


class DDT:
    """
    The DDT dataset has been annotated with NER tags in the IOB2 format.

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
        import pyconll

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
        corpus: Corpus = ColumnCorpus(self.dataset_dir, columns,
                                      train_file='{}.{}{}'.format(self.dataset_name, 'train', self.file_extension),
                                      test_file='{}.{}{}'.format(self.dataset_name, 'test', self.file_extension),
                                      dev_file='{}.{}{}'.format(self.dataset_name, 'dev', self.file_extension))

        # Remove the `name=` from `name=B-PER` to only use the `B-PER` tag
        parts = ['train', 'dev', 'test']
        for part in parts:
            dataset = corpus.__getattribute__(part)

            for sentence in dataset.sentences:
                for token in sentence.tokens:
                    token.tags['ner'].value = token.tags['ner'].value.split("=")[1].replace("|SpaceAfter", "")

        return corpus
