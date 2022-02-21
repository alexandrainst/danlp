import os
import pandas as pd

from danlp.download import DEFAULT_CACHE_DIR, download_dataset, _unzip_process_func, DATASETS

class DDisco:
    """

    Class for loading the DDisco dataset.
    The DDisco dataset is annotated for discourse coherence. 
    It contains user-generated texts from Reddit and Wikipedia.

    Annotation labels are: 
    * 1: low coherence
    * 2: medium coherence
    * 3: high coherence

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity

    """
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.dataset_name = 'ddisco'
        self.file_extension = DATASETS[self.dataset_name]['file_extension']
        self.dataset_dir = download_dataset(self.dataset_name, process_func=_unzip_process_func, cache_dir=cache_dir)

    def load_with_pandas(self):
        """
        Loads the DDisco dataset in dataframes with pandas. 
        
        :return: 2 dataframes -- train, test
        """
        df_train = pd.read_csv(os.path.join(self.dataset_dir, self.dataset_name + '.train' + self.file_extension), sep='\t', index_col=0, encoding='utf-8').dropna()
        df_test = pd.read_csv(os.path.join(self.dataset_dir, self.dataset_name + '.test' + self.file_extension), sep='\t', index_col=0, encoding='utf-8').dropna()
        
        return df_train, df_test
