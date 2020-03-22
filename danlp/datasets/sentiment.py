import os

import pandas as pd

from danlp.download import DATASETS, download_dataset, DEFAULT_CACHE_DIR


class EuroparlSentiment:
    
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.dataset_name = 'europarl.sentiment'
        self.file_extension = DATASETS[self.dataset_name]['file_extension']

        self.dataset_dir = download_dataset(self.dataset_name, cache_dir=cache_dir)
        self.file_path = os.path.join(self.dataset_dir, self.dataset_name + self.file_extension)
        
    def load_with_pandas(self):
        """ Load and drop duplicates and nan values"""

        df = pd.read_csv(self.file_path, sep=',', index_col=0, encoding='utf-8')

        df = df[['valence', 'text']].dropna()
        return df.drop_duplicates()


class LccSentiment:
    
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.dataset_name1 = 'lcc1.sentiment'
        self.file_extension1 = DATASETS[self.dataset_name1]['file_extension']

        self.dataset_dir1 = download_dataset(self.dataset_name1, cache_dir=cache_dir)
        self.file_path1 = os.path.join(self.dataset_dir1, self.dataset_name1 + self.file_extension1)
        
        self.dataset_name2 = 'lcc2.sentiment'
        self.file_extension2 = DATASETS[self.dataset_name2]['file_extension']

        self.dataset_dir2 = download_dataset(self.dataset_name2, cache_dir=cache_dir)
        self.file_path2 = os.path.join(self.dataset_dir2, self.dataset_name2 + self.file_extension2)
        
    def load_with_pandas(self):
        """ Load, combine and drop duplicates and nan values """
        
        df1 = pd.read_csv(self.file_path1, sep=',', encoding='utf-8')
        df2 = pd.read_csv(self.file_path2, sep=',', encoding='utf-8')
       
        df = df1.append(df2, sort=False)
        df = df[['valence', 'text']].dropna()
        
        return df

    
class TwitterSent:
    
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.dataset_name = 'twitter_sentiment'
        #self.file_extension = DATASETS[self.dataset_name]['file_extension']
        
        self.dataset_dir = download_dataset(self.dataset_name, cache_dir=cache_dir)
        #self.file_path = os.path.join(self.dataset_dir, self.dataset_name + '.csv')
        
    #def load_with_pandas(self):
    #    return pd.read_csv(self.file_path, sep=',', encoding='utf-8')