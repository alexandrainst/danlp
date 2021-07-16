import hashlib
import inspect
import os
import urllib
from pathlib import Path
from typing import Callable
from tempfile import NamedTemporaryFile

from tqdm import tqdm

from danlp.utils import extract_single_file_from_zip

DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), '.danlp')

DANLP_STORAGE_URL = 'http://danlp-downloads.alexandra.dk'

# The naming convention of the word embedding are on the form <dataset>.<lang>.<type>
# The <type> can be subword vectors=swv or word vectors=wv
MODELS = {
    # WORD EMBEDDINGS
    'wiki.da.wv': {
        'url': 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.da.vec',
        'vocab_size': 312956,
        'dimensions': 300,
        'md5_checksum': '892ac16ff0c730d7230c82ad3d565984',
        'size': 822569731,
        'file_extension': '.bin'
    },
    'cc.da.wv': {
        'url': 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.da.300.vec.gz',
        'vocab_size': 2000000,
        'dimensions': 300,
        'md5_checksum': '68a766bf25409ae96d334493df026419',
        'size': 1221429539,
        'file_extension': '.bin'
    },
    'conll17.da.wv': {
        'url': 'http://vectors.nlpl.eu/repository/11/38.zip',
        'vocab_size': 1655870,
        'dimensions': 100,
        'md5_checksum': 'cc324d04a429f80825fede0d6502543d',
        'size': 624863834,
        'file_extension': '.bin'
    },
    'news.da.wv': {
        'url': 'https://loar.kb.dk/bitstream/handle/1902/329/danish_newspapers_1880To2013.txt?sequence=4&isAllowed=y',
        'vocab_size': 2404836,
        'dimensions': 300,
        'size': 6869762980,
        'md5_checksum': 'e0766f997e04dddf65aec5e2691bf36d',
        'file_extension': '.bin'
    },
    'dslreddit.da.wv': {
        'url': 'https://ndownloader.figshare.com/files/15111116',
        'file_extension': '.bin',
        'dimensions': 300,
        'vocab_size': 178649,
        'size': 202980709,
        'md5_checksum': '6846134374cfae008a32f62dea5ed8bf',
    },
    'sketchengine.da.wv': {
        'url': 'https://embeddings.sketchengine.co.uk/static/models/word/datenten14_5.vec',
        'dimensions': 100,
        'vocab_size': 2360830,
        'size': 2053148194,
        'md5_checksum': '80cced3e135274d2815f55ca2a7eafcd',
        'file_extension': '.bin'
    },
    'sketchengine.da.swv': {
        'url': 'https://embeddings.sketchengine.co.uk/static/models/word/datenten14_5.bin',
        'size': 2739302263,
        'md5_checksum': '7387bfa5be6fbf525734b617e3d547de',
        'file_extension': '.bin'
    },
    'wiki.da.swv': {
        'url': 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.da.zip',
        'md5_checksum': '86e7875d880dc1f4d3e7600a6ce4952d',
        'size': 3283027968,
        'file_extension': '.bin'
    },
    'cc.da.swv': {
        'url': 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.da.300.bin.gz',
        'size': 4509731789,
        'md5_checksum': '562d7b49ab8ee45892f6e28b02db5f01',
        'file_extension': '.bin'
    },

    # CONTEXTUAL EMBEDDINGS
    'flair.fwd': {
        'url': DANLP_STORAGE_URL+'/models/flair.fwd.zip',
        'md5_checksum': '8697e286048a4aa30acc62995397a0c8',
        'size': 18548086,
        'file_extension': '.pt'
    },
    'flair.bwd': {
        'url': DANLP_STORAGE_URL+'/models/flair.bwd.zip',
        'md5_checksum': '11549f1dc28f92a7c37bf511b023b1f1',
        'size': 18551173,
        'file_extension': '.pt'
    },

    # POS MODELS
    'flair.pos': {
        'url': DANLP_STORAGE_URL + '/models/flair.pos.zip',
        'md5_checksum': '627321171ecf4f7933b5e10602a60cbe',
        'size': 424727006,
        'file_extension': '.pt'
    },
    
    # NER MODELS
    'flair.ner': {
        'url': DANLP_STORAGE_URL + '/models/flair_ner.zip',
        'md5_checksum': 'a1cf475659d1cf3a0f5eae5377f7027e',
        'size': 419047115,
        'file_extension': '.pt'
    },
    'bert.ner': {
        'url': DANLP_STORAGE_URL + '/models/bert.ner.zip',
        'md5_checksum': '8929acefcbc4b2819d0ee88fa1a79011',
        'size': 1138686784,
        'file_extension': ''
    },

    # Spacy MODELS
    'spacy': {
        'url': DANLP_STORAGE_URL + '/models/spacy.zip',
        'md5_checksum': '43de8cadab206234537b04a4cca24e71',
        'size': 1261762677,
        'file_extension': ''
    },
    'spacy.sentiment': {
        'url': DANLP_STORAGE_URL + '/models/spacy.sentiment.zip',
        'md5_checksum': 'bfe1fcd4a821b3dcc1a23a36497cc6c8',
        'size': 752316341,
        'file_extension': ''
    },

    # BERT models 
    'bert.botxo.pytorch': {
        'url': DANLP_STORAGE_URL + '/models/bert.botxo.pytorch.zip',
        'md5_checksum': '8d73bb1ad154c3ca137ab06a3d5d37a1',
        'size': 413164986,
        'file_extension': ''
    },
    'bert.emotion': {
        'url': DANLP_STORAGE_URL + '/models/bert.emotion.zip',
        'md5_checksum': '832214e9362b12372bedbbc8e819ea9d',
        'size': 410902634,
        'file_extension': ''
    },
    'bert.noemotion': {
        'url': DANLP_STORAGE_URL + '/models/bert.noemotion.zip',
        'md5_checksum': 'e5ad6ebc0cfb3cd65677aa524c75b8c9',
        'size': 410883271,
        'file_extension': ''
    },
    'bert.subjective': {
        'url': DANLP_STORAGE_URL + '/models/bert.sub.v0.0.1.zip',
        'md5_checksum': 'b713a8ec70ca4e8269d7a66a1cda2366',
        'size': 410882683,
        'file_extension': ''
    },
    'bert.polarity': {
        'url': DANLP_STORAGE_URL + '/models/bert.pol.v0.0.1.zip',
        'md5_checksum': 'bb0940fba75b39795332105bb2bc2af1',
        'size': 410888897,
        'file_extension': ''
    },
    'bert.offensive': {
        'url': DANLP_STORAGE_URL + '/models/bert.offensive.zip',
        'md5_checksum': '48786f272321e42ad8b2bb5ee4142f5c',
        'size': 410733108,
        'file_extension': ''
    },
    # XLMR models
    'xlmr.coref': {
        'url': DANLP_STORAGE_URL + '/models/xlmr.coref.zip',
        'md5_checksum': '7cb9032c6b3a6af9d22f372de5817b35',
        'size': 853720929,
        'file_extension': '.tar.gz'
    },
}

DATASETS = {
    'ddt': {
        'url': DANLP_STORAGE_URL + '/datasets/ddt.zip',
        'md5_checksum': 'b087c3a525f1cdc868b3f0a2e437a04d',
        'size': 1209710,
        'file_extension': '.conllu'
    },
    'wikiann': {
        'url': 'https://blender04.cs.rpi.edu/~panx2/wikiann/data/da.tar.gz',
        'md5_checksum': 'e23d0866111f9980bbc7421ee3124deb',
        'size': 4458532,
        'file_extension': '.iob'
    },
    'wordsim353.da': {
        'url': 'https://raw.githubusercontent.com/fnielsen/dasem/master/dasem/data/wordsim353-da/combined.csv',
        'md5_checksum': '7ac76acba4af2d90c04136bc6b227e54',
        'size': 12772,
        'file_extension': '.csv'
    },
    'dsd': {
        'url': 'https://raw.githubusercontent.com/kuhumcst/Danish-Similarity-Dataset/master/gold_sims_da.csv',
        'md5_checksum': '5e7dad9e6c8c32aa9dd17830bed5e0f6',
        'size': 3489,
        'file_extension': '.csv'
    },
    # coreference dataset 
    'dacoref': {
        'url': 'http://danlp-downloads.alexandra.dk/datasets/dacoref.zip',
        'md5_checksum': 'e6f2707f4f600a0d357dc7afa1b01f92',
        'size': 1005278,
        'file_extension': ''
    },
    # DKHate 
    'dkhate': {
        'url': 'http://danlp-downloads.alexandra.dk/datasets/dkhate.zip',
        'md5_checksum': '79d5abc0492637701edd0ca08154eb58',
        'size': 170933,
        'file_extension': '.tsv'
    },
    # Danish Wordnet
    'dannet': {
        'url': DANLP_STORAGE_URL + '/datasets/dannet.zip',
        'md5_checksum': 'a5aa388bb08487bd59d72257aa15d8fa',
        'size': 6083044,
        'file_extension': '.csv'
    },
    # Danish Unimorph
    'unimorph': {
        'url': DANLP_STORAGE_URL + '/datasets/unimorph.zip',
        'md5_checksum': '88d086c8b69523d8387c464bf0f82d7a',
        'size': 99725,
        'file_extension': '.tsv'
    },
    
    # SENTIMENT EVALUATION
    'europarl.sentiment1': {
        'url': 'https://raw.githubusercontent.com/fnielsen/europarl-da-sentiment/master/europarl-da-sentiment.csv',
        'md5_checksum': 'eb12513f04ead1dc0b455e738bf8d831',
        'size': 3620027,
        'file_extension': '.csv'
    },
    'lcc1.sentiment': {
        'url': 'https://raw.githubusercontent.com/fnielsen/lcc-sentiment/master/dan_mixed_2014_10K-sentences.csv',
        'md5_checksum': 'd1b19d2aa53b4d598ffd8ca35750dd43',
        'size': 1202967,
        'file_extension': '.csv'
    },
    'lcc2.sentiment': {
        'url': 'https://raw.githubusercontent.com/fnielsen/lcc-sentiment/master/dan_newscrawl_2011_10K-sentences.csv',
        'md5_checksum': '4a311472bad5b934c45015d2359dfce6',
        'size': 1386727,
        'file_extension': '.csv'
    },
    'twitter.sentiment': {
        'url': DANLP_STORAGE_URL + '/datasets/twitter.sentiment.zip',
        'md5_checksum': 'b12633e3f55b69e7a6981ff0017c01e5', 
        'size': 17365, 
        'file_extension': '.csv'
    },
    'europarl.sentiment2': {
        'url': DANLP_STORAGE_URL + '/datasets/europarl.sentiment2.zip',
        'md5_checksum': 'fa7cfc829867a00dba74d917c78df294', 
        'size': 51545, 
        'file_extension': '.csv'
    },
    'angrytweets.sentiment': {
        'url': DANLP_STORAGE_URL + '/datasets/game_tweets.zip',
        'md5_checksum': '9451ea83333efb98317d2e3267ab8508', 
        'size': 57903,
        'file_extension': '.csv'
    },
}


class TqdmUpTo(tqdm):
    """
    This class provides callbacks/hooks in order to use tqdm with urllib.
    Read more here:
    https://github.com/tqdm/tqdm#hooks-and-callbacks
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_dataset(dataset_name: str, cache_dir: str = DEFAULT_CACHE_DIR,
                     process_func: Callable = None, verbose: bool = False, force_download: bool = False):
    """
    :param str dataset_name:
    :param str cache_dir:
    :param process_func:
    :param bool verbose:
    :param bool force_download:
    :return:
    """
    if dataset_name not in DATASETS:
        raise ValueError("The dataset {} do not exist".format(dataset_name))

    dataset_dir = os.path.join(cache_dir, dataset_name)
    dataset_info = DATASETS[dataset_name]
    dataset_info['name'] = dataset_name

    if not os.path.isdir(dataset_dir) or not os.listdir(dataset_dir) or force_download:  # Then dataset has not been downloaded
        os.makedirs(dataset_dir, exist_ok=True)

        _download_and_process(dataset_info, process_func, dataset_dir, verbose)

    else:
        if verbose:
            print("Dataset {} exists in {}".format(dataset_name, dataset_dir))

    return dataset_dir


def download_model(model_name: str, cache_dir: str = DEFAULT_CACHE_DIR, process_func: Callable = None,
                   verbose: bool = False, clean_up_raw_data=True, force_download: bool = False, file_extension=None):
    """
    :param str model_name:
    :param str cache_dir: the directory for storing cached data
    :param process_func:
    :param bool verbose:
    :param bool clean_up_raw_data:
    :param bool force_download:
    :param str file_extension:
    """
    if model_name not in MODELS:
        raise ValueError("The model {} do not exist".format(model_name))

    model_info = MODELS[model_name]
    model_info['name'] = model_name

    model_file = model_name + model_info['file_extension'] if not file_extension else model_name + file_extension
    model_file_path = os.path.join(cache_dir, model_file)

    if not os.path.exists(model_file_path) or force_download:
        os.makedirs(cache_dir, exist_ok=True)

        _download_and_process(model_info, process_func, model_file_path, verbose)

    else:
        if verbose:
            print("Model {} exists in {}".format(model_name, model_file_path))

    return model_file_path


def _check_file(fname):
    """
    Method borrowed from
    https://github.com/fastai/fastai/blob/master/fastai/datasets.py
    :param fname:
    :return:
    """
    size = os.path.getsize(fname)
    with open(fname, "rb") as f:
        hash_nb = hashlib.md5(f.read(2 ** 20)).hexdigest()
    return size, hash_nb


def _check_process_func(process_func: Callable):
    """
    Checks that a process function takes the correct arguments
    :param process_func:
    """
    function_args = inspect.getfullargspec(process_func).args
    expected_args = ['tmp_file_path', 'meta_info', 'cache_dir', 'clean_up_raw_data', 'verbose']

    assert function_args[:len(expected_args)] == expected_args, "{} does not have the correct arguments".format(process_func)


def _download_and_process(meta_info: dict, process_func: Callable, single_file_path, verbose):
    """
    :param meta_info:
    :param process_func:
    :param single_file_path:
    :param verbose:
    """

    if process_func is not None:

        _check_process_func(process_func)

        tmp_file_path = NamedTemporaryFile().name
        _download_file(meta_info, tmp_file_path, verbose=verbose)

        cache_dir = os.path.split(single_file_path)[0]
        process_func(tmp_file_path, meta_info, cache_dir=cache_dir, verbose=verbose, clean_up_raw_data=True)

    else:
        single_file = meta_info['name'] + meta_info['file_extension']
        _download_file(meta_info, os.path.join(single_file_path, single_file), verbose=verbose)


def _download_file(meta_info: dict, destination: str, verbose: bool = False):
    """
    :param meta_info:
    :param destination:
    :param verbose:
    """
    file_name = os.path.split(destination)[1]

    expected_size = meta_info['size']
    expected_hash = meta_info['md5_checksum']
    url = meta_info['url']

    if not os.path.isfile(destination):
        if verbose:
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1) as t:
                t.set_description("Downloading file {}".format(destination))
                urllib.request.urlretrieve(url, destination, reporthook=t.update_to)
        else:
            print("Downloading file {}".format(destination))
            urllib.request.urlretrieve(url, destination)

    else:
        if verbose:
            print("The file {} exists here: {}".format(file_name, destination))
            
    assert _check_file(destination) == (expected_size, expected_hash), \
        "Downloaded file does not match the expected size or checksum! Remove the file: {} and try again.".format(destination)


def _unzip_process_func(tmp_file_path: str, meta_info: dict, cache_dir: str = DEFAULT_CACHE_DIR,
                        clean_up_raw_data: bool = True, verbose: bool = False, file_in_zip: str = None):
    """
    Simple process function for processing models
    that only needs to be unzipped after download.

    :param str tmp_file_path: The path to the downloaded raw file
    :param dict meta_info:
    :param str cache_dir:
    :param bool clean_up_raw_data:
    :param bool verbose:
    :param str file_in_zip: Name of the model file in the zip, if the zip contains more than one file

    """
    from zipfile import ZipFile
    
    model_name = meta_info['name']
    full_path = os.path.join(cache_dir, model_name) + meta_info['file_extension']

    if verbose:
        print("Unzipping {} ".format(model_name))

    with ZipFile(tmp_file_path, 'r') as zip_file:  # Extract files to cache_dir
        
        file_list = zip_file.namelist()

        if len(file_list) == 1:
            extract_single_file_from_zip(cache_dir, file_list[0], full_path, zip_file)

        elif file_in_zip:
            extract_single_file_from_zip(cache_dir, file_in_zip, full_path, zip_file)

        else:  # Extract all the files to the name of the model/dataset
            destination = os.path.join(cache_dir, meta_info['name'])
            zip_file.extractall(path=destination)
