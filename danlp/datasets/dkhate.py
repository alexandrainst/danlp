import os
import pandas as pd

from danlp.download import DEFAULT_CACHE_DIR, download_dataset, _unzip_process_func, DATASETS


class DKHate:
    """

    Class for loading the DKHate dataset.
    The DKHate dataset contains user-generated comments from social media platforms (Facebook and Reddit) 
    annotated for various types and target of offensive language. 
    The original corpus has been used for the OffensEval 2020 shared task.
    Note that only labels for Offensive language identification (sub-task A) are available.
    Which means that each sample in this dataset is labelled with either `NOT` (Not Offensive) or `OFF` (Offensive).

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity

    """
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.dataset_name = 'dkhate'
        self.file_extension = DATASETS[self.dataset_name]['file_extension']
        self.dataset_dir = download_dataset(self.dataset_name, process_func=_unzip_process_func, cache_dir=cache_dir)

    def load_with_pandas(self):
        """
        Loads the DKHate dataset in a dataframe with pandas. 
        
        :return: a dataframe for test data and a dataframe from train data
        """
        df_test = pd.read_csv(os.path.join(self.dataset_dir, self.dataset_name + '.test' + self.file_extension), sep='\t', index_col=0, encoding='utf-8').dropna()
        df_train = pd.read_csv(os.path.join(self.dataset_dir, self.dataset_name + '.train' + self.file_extension), sep='\t', index_col=0, encoding='utf-8').dropna()
        
        return df_test, df_train