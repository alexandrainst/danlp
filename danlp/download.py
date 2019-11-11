import hashlib
import inspect
import os
import shutil
import urllib
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from danlp.utils import random_string

DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), '.danlp')

DANLP_S3_URL = 'https://danlp.s3.eu-central-1.amazonaws.com'

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
    'sketchengine.da.wv': {
        'url': 'https://embeddings.sketchengine.co.uk/static/models/lc/datenten14_5.vec',
        'dimensions': 100,
        'vocab_size': 2360830,
        'size': 2053148194,
        'md5_checksum': '80cced3e135274d2815f55ca2a7eafcd',
        'file_extension': '.bin'
    },
    'sketchengine.da.swv': {
        'url': 'https://embeddings.sketchengine.co.uk/static/models/lc/datenten14_5.bin',
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
        'url': DANLP_S3_URL+'/models/flair.fwd.zip',
        'md5_checksum': '8697e286048a4aa30acc62995397a0c8',
        'size': 18548086,
        'file_extension': '.pt'
    },
    'flair.bwd': {
        'url': DANLP_S3_URL+'/models/flair.bwd.zip',
        'md5_checksum': '11549f1dc28f92a7c37bf511b023b1f1',
        'size': 18551173,
        'file_extension': '.pt'
    },

    # POS MODELS
    'flair.pos': {
        'url': DANLP_S3_URL + '/models/flair.pos.zip',
        'md5_checksum': 'b9892d4c1c654503dff7e0094834d6ed',
        'size': 426404955,
        'file_extension': '.pt'
    },
    
    # NER MODELS
    'flair.ner': {
        'url': DANLP_S3_URL + '/models/flair.ner.zip',
        'md5_checksum': 'a1cf475659d1cf3a0f5eae5377f7027e',
        'size': 419047115,
        'file_extension': '.pt'
    }
}

DATASETS = {
    'ddt': {
        'url': DANLP_S3_URL + '/datasets/ddt.zip',
        'md5_checksum': '7bdd5d4f43dd9c4de35a48ea58f950ca',
        'size': 1205299,
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
    }
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


def download_dataset(dataset: str, cache_dir: str = DEFAULT_CACHE_DIR,
                     process_func: Callable = None, verbose: bool = False):
    """

    :param verbose:
    :param dataset:
    :param cache_dir:
    :param process_func:
    :return:
    """
    if dataset not in DATASETS:
        raise ValueError("The dataset {} do not exist".format(dataset))

    dataset_dir = os.path.join(cache_dir, dataset)
    dataset_info = DATASETS[dataset]
    dataset_info['name'] = dataset

    if not os.path.isdir(dataset_dir):  # Then dataset has not been downloaded
        os.makedirs(dataset_dir, exist_ok=True)

        file_path = os.path.join(cache_dir, dataset)

        _download_and_process(dataset_info, process_func, file_path, verbose)

    else:
        if verbose:
            print("Dataset {} exists in {}".format(dataset, dataset_dir))

    return dataset_dir


def download_model(model_name: str, cache_dir: str = DEFAULT_CACHE_DIR, process_func: Callable = None,
                   verbose: bool = False, clean_up_raw_data=True, force_download: bool = False, file_extension=None):
    """
    :param file_extension:
    :param force_download:
    :param model_name:
    :param process_func:
    :param bool clean_up_raw_data:
    :param str cache_dir: the directory for storing cached data
    :param bool verbose:
    """
    if model_name not in MODELS:
        raise ValueError("The model {} do not exist".format(model_name))

    model_info = MODELS[model_name]
    model_info['name'] = model_name

    model_file = model_name + model_info['file_extension'] if not file_extension else model_name + file_extension
    model_file_path = os.path.join(cache_dir, model_file)

    if not os.path.isfile(model_file_path) or force_download:
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

        tmp_file_path = "/tmp/{}.tmp".format(random_string())

        _download_file(meta_info, tmp_file_path, verbose=verbose)

        process_func(tmp_file_path, meta_info, verbose=verbose, clean_up_raw_data=True)

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
        "Downloaded file does not match the expected checksum! Remove the file: {} and try again.".format(destination)


def _unzip_process_func(tmp_file_path: str, meta_info: dict, cache_dir: str = DEFAULT_CACHE_DIR,
                        clean_up_raw_data: bool = True, verbose: bool = False, file_in_zip: str = None):
    """
    Simple process function for processing models
    that only needs to be unzipped after download.

    :param tmp_file_path: The path to the downloaded raw file
    :param clean_up_raw_data:
    :param verbose:
    :param file_in_zip: Name of the model file in the zip, if the zip contains more than one file
    """
    from zipfile import ZipFile

    model_name = meta_info['name']

    full_path = os.path.join(cache_dir, model_name) + meta_info['file_extension']

    if verbose:
        print("Unzipping raw {} embeddings".format(model_name))

    with ZipFile(tmp_file_path, 'r') as zip_file:  # Extract files to cache_dir

        file_list = zip_file.namelist()

        if len(file_list) == 1:
            _extract_single_file_from_zip(cache_dir, file_list[0], full_path, zip_file)

        elif file_in_zip:
            _extract_single_file_from_zip(cache_dir, file_in_zip, full_path, zip_file)

        else:  # Extract all the files to the name of the model/dataset
            destination = os.path.join(cache_dir, meta_info['name'])
            zip_file.extractall(path=destination)


def _extract_single_file_from_zip(cache_dir: str, file_in_zip: str, full_path, zip_file):
    # To not have name conflicts

    tmp_path = os.path.join(cache_dir, ''.join(random_string()))

    outpath = zip_file.extract(file_in_zip, path=tmp_path)
    os.rename(outpath, full_path)

    shutil.rmtree(tmp_path)


