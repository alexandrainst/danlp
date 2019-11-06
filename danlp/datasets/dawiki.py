import os
import subprocess
import shlex

from danlp.download import download_dataset, DEFAULT_CACHE_DIR

class DA_WIKI():
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR, verbose: bool =
            False):
        self.cache_dir = cache_dir
        self.verbose = verbose

    def _download(self):
        if self.verbose:
            subprocess.call('bash wiki_downloader.sh {}'.format(self.cache_dir), shell=True)
        else:
            # run script silent
            subprocess.call('bash wiki_downloader.sh {}'.format(self.cache_dir), shell=True, stdout=open(os.devnull, 'wb'))

    def _load(self):
        with open(os.path.join(self.cache_dir,"dawiki","dawiki.txt")) as txtfile:
            data = txtfile.read()
        return data

    def load_as_txt(self):
        self._download()
        return self._load()
