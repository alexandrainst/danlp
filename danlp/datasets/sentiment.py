import os

import pandas as pd

from danlp.download import DATASETS, download_dataset, DEFAULT_CACHE_DIR

class EuroParlSent:
    
    
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.dataset_name = 'europarl_sentiment'
        self.file_extension = DATASETS[self.dataset_name]['file_extension']

        self.dataset_dir = download_dataset(self.dataset_name, cache_dir=cache_dir)
        self.file_path = os.path.join(self.dataset_dir, self.dataset_name + self.file_extension)
        
    def load_with_pandas(self):
        "load and drop duplicates and nan values"
        
        df = pd.read_csv(self.file_path, sep=',',index_col=0,encoding='utf-8')
       
        df=df[['valence','text']].dropna()
        return df.drop_duplicates()
    
class TwitterSent:
    
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.dataset_name = 'twitter_sentiment'
        #self.file_extension = DATASETS[self.dataset_name]['file_extension']
        
        self.dataset_dir = download_dataset(self.dataset_name, cache_dir=cache_dir)
        #self.file_path = os.path.join(self.dataset_dir, self.dataset_name + '.csv')
        
    #def load_with_pandas(self):
    #    return pd.read_csv(self.file_path, sep=',', encoding='utf-8')