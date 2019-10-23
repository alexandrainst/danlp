import os
import subprocess
import shlex

from danlp.datasets.urls.opus_urls import OPUS_MONO_DA
from danlp.download import download_dataset, DEFAULT_CACHE_DIR

class DA_WIKI():
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR, verbose: bool =
            False):
        self.cache_dir = cache_dir
        self.verbose = verbose

    def _download(self):
        subprocess.call(shlex.split('./wiki_downloader.sh {}'.format(self.cache_dir)))

    def _load(self):
        with open(os.path.join(self.cache_dir,"dawiki","dawiki.txt")) as txtfile:
            data = txtfile.read()
        return data

dawiki = DA_WIKI(verbose=True)

dawiki._download()

print(dawiki._load()[0:100])
