import os

import pandas as pd
import tweepy
from tweepy import TweepError

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
    
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR, force: bool =False):
        self.dataset_name = 'twitter.sentiment'
        #self.file_extension = DATASETS[self.dataset_name]['file_extension']

        self.dataset_dir = download_dataset(self.dataset_name, cache_dir=cache_dir, process_func=_twitter_data_process_func, force=force)
        self.file_path = os.path.join(self.dataset_dir, self.dataset_name + '.csv')
        
    def load_with_pandas(self):
        df=pd.read_csv(self.file_path, sep=',', encoding='utf-8')
        return df[df['part'] == 'test'].drop(columns=['part']), df[df['part'] == 'train'].drop(columns=['part'])


def lookup_tweets(tweet_ids, api):
    full_tweets = []
    tweet_count = len(tweet_ids)
    try:
        for i in range(int(tweet_count/100)+1):
            # Catch the last group if it is less than 100 tweets
            end_loc = min((i + 1) * 100, tweet_count)
            full_tweets.extend(
                api.statuses_lookup(id_=tweet_ids[i * 100:end_loc], tweet_mode='extended', trim_user=True)
            )
        return full_tweets
    except tweepy.TweepError:
        print("Failed fetching tweets")


def _twitter_data_process_func(tmp_file_path: str, meta_info: dict,
                               cache_dir: str = DEFAULT_CACHE_DIR,
                               clean_up_raw_data: bool = True,
                               verbose: bool = True):
    from zipfile import ZipFile

    twitter_api = construct_twitter_api_connection()

    with ZipFile(tmp_file_path, 'r') as zip_file:  # Extract files to cache_dir
        ids_file_path = zip_file.extract('twitter.sentiment.csv',
                                         path=os.path.join(cache_dir,
                                                           meta_info['name']))

    df = pd.read_csv(ids_file_path)

    twitter_ids = list(df['twitterid'])

    full_t = lookup_tweets(twitter_ids, twitter_api)
    tweet_texts = [[tweet.id, tweet.full_text] for tweet in full_t]
    tweet_ids, t_texts = list(zip(*tweet_texts))
    tweet_texts_df = pd.DataFrame({'twitterid': tweet_ids, 'text': t_texts})

    resulting_df = pd.merge(df, tweet_texts_df)
    dataset_path = os.path.join(cache_dir, meta_info['name'],
                                meta_info['name'] + meta_info[
                                    'file_extension'])

    resulting_df.to_csv(dataset_path, index=False)
    
    if verbose:
        print("Downloaded {} out of {} tweets".format(len(full_t), len(twitter_ids)))
 

def construct_twitter_api_connection():
    if not('TWITTER_CONSUMER_KEY' in os.environ
           and 'TWITTER_CONSUMER_SECRET' in os.environ
           and 'TWITTER_ACCESS_TOKEN' in os.environ
           and 'TWITTER_ACCESS_SECRET' in os.environ):
        exit("The Twitter API keys was not found."
              "\nTo download tweets you need to set the following environment "
              "variables: \n- 'TWITTER_CONSUMER_KEY'\n- 'TWITTER_CONSUMER_SECRET'"
              "\n- 'TWITTER_ACCESS_TOKEN'\n- 'TWITTER_ACCESS_SECRET' "
              "\n\nThe keys can be obtained from "
              "https://developer.twitter.com")

    twitter_consumer_key = os.environ.get('TWITTER_CONSUMER_KEY')
    twitter_consumer_secret = os.environ.get('TWITTER_CONSUMER_SECRET')
    twitter_access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
    twitter_access_secret = os.environ.get('TWITTER_ACCESS_SECRET')

    auth = tweepy.OAuthHandler(twitter_consumer_key, twitter_consumer_secret)
    auth.set_access_token(twitter_access_token, twitter_access_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    try:
        api.verify_credentials()
    except TweepError:
        exit("Could not establish connection to the Twitter API, have you provieded the correct keys?")

    return api