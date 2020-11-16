import os

import pandas as pd

from danlp.download import DATASETS, download_dataset, DEFAULT_CACHE_DIR


class WordSim353Da:
    """

    Class for loading the WordSim-353 dataset.

    :param str cache_dir: the directory for storing cached models

    """
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.dataset_name = 'wordsim353.da'
        self.file_extension = DATASETS[self.dataset_name]['file_extension']

        self.dataset_dir = download_dataset(self.dataset_name, process_func=_word_sim_process_func, cache_dir=cache_dir)
        self.file_path = os.path.join(self.dataset_dir, self.dataset_name + self.file_extension)

    def load_with_pandas(self):
        """
        Loads the dataset in a dataframe.

        :return: a dataframe
        """
        return pd.read_csv(self.file_path)

    def words(self) -> set:
        """
        Loads the vocabulary.

        :rtype: set
        """
        df = self.load_with_pandas()
        return set(df['da1']) | set(df['da2'])


def _word_sim_process_func(tmp_file_path: str, meta_info: dict, cache_dir: str = DEFAULT_CACHE_DIR,
                          clean_up_raw_data: bool = True, verbose: bool = False):

    df = pd.read_csv(tmp_file_path)
    del df['Word 1']
    del df['Word 2']
    del df['Problem']

    file_path = os.path.join(cache_dir, meta_info['name'], meta_info['name'] + meta_info['file_extension'])
    df.to_csv(file_path, index=False)
    os.remove(tmp_file_path)


class DSD:
    """

    Class for loading the Danish Similarity Dataset dataset.

    :param str cache_dir: the directory for storing cached models

    """
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.dataset_name = 'dsd'
        self.file_extension = DATASETS[self.dataset_name]['file_extension']

        self.dataset_dir = download_dataset(self.dataset_name, cache_dir=cache_dir)
        self.file_path = os.path.join(self.dataset_dir, self.dataset_name + self.file_extension)

    def load_with_pandas(self):
        """
        Loads the dataset in a dataframe.

        :return: a dataframe
        """
        return pd.read_csv(self.file_path, delimiter="\t")

    def words(self) -> set:
        """
        Loads the vocabulary.

        :rtype: set
        """
        df = self.load_with_pandas()
        return set(df['word1']) | set(df['word2'])
