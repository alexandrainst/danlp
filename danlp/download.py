from tqdm import tqdm
from pathlib import Path
import hashlib
import os
import urllib

from typing import Callable

DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), '.danlp')

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
    'connl.da.wv': {
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
        'url': 'https://nextcloud.alexandra.dk/index.php/s/oYreE92iLW4Bj4D/download',
        'md5_checksum': 'a2141e29c401543e5aa6e45c3ce69d62',
        'size': 19974935,
        'file_extension': '.pt'
    },
    'flair.bwd': {
        'url': 'https://nextcloud.alexandra.dk/index.php/s/DYeJxpk3Efwnknj/download',
        'md5_checksum': '888fdbc1283dbb9419e96d93da7ea57b',
        'size': 19974935,
        'file_extension': '.pt'
    },

    # POS MODELS
    'flair.pos': {
        'url': 'https://nextcloud.alexandra.dk/index.php/s/gos5i3DgkZGNYjE/download',
        'md5_checksum': '913cb7956d49d7f0b3b3827608086ac4',
        'size': 476407374, 
        'file_extension': '.pt'
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


def download_model(model_name: str, cache_dir: str = DEFAULT_CACHE_DIR, process_func: Callable = None,
                    verbose: bool = False, clean_up_raw_data=True, force_download: bool = False):
    """
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

    model_file = model_name + model_info['file_extension']
    model_file_path = os.path.join(cache_dir, model_file)

    if not os.path.isfile(model_file_path) or force_download:
        os.makedirs(cache_dir, exist_ok=True)

        url = model_info['url']
        expected_size = model_info['size']
        expected_hash = model_info['md5_checksum']

        if process_func is not None:
            # A temporary file is downloaded which will be processed by the process func
            tmp_dl_file = model_name + ".tmp"
            tmp_file_path = os.path.join(cache_dir, tmp_dl_file)

            _download_file(url, tmp_file_path, expected_size, expected_hash, verbose=verbose)

            process_func(tmp_file_path, verbose=verbose, clean_up_raw_data=clean_up_raw_data)

        else:
            # The model file will be downloaded directly to the model_file_path
            _download_file(url, model_file_path, expected_size, expected_hash)

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
        hash_nb = hashlib.md5(f.read(2**20)).hexdigest()
    return size, hash_nb


def _download_file(url: str, destination: str, expected_size: int, expected_hash: str, verbose: bool = False):
    file_name = os.path.split(destination)[1]

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
