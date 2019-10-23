import os
from danlp.datasets.urls.opus_urls import OPUS_MONO_DA
from danlp.download import download_dataset, DEFAULT_CACHE_DIR
import re

class OPUS():
    """
    Dataloader and downloading class OPUS datasets (see http://opus.nlpl.eu/).
    Datasets includes OpenSubtitles2018, EUBookshop2, EuroParl8, ParacCrawl5,
    DGT2019, ECB1 and others (see urls/opus_mono_da.json).

    NOTE:
        Currently only files preprocesses as 'mono' is available.

    OPUS keeps its data in different formats
    (see http://opus.nlpl.eu/trac/wiki/DataFormats).
    It tries to be consistent in its data formats.
    Dependending on the formats of the files, different processing steps
    should be performed. The format of the files is specified in the
    'preprocessing' value of the corpora descriptions.

    - 'mono'
        Single file of text without markup and with sentence delimiter. Has unprocessed and
        tokenized versions, specified by the url.

    - 'raw'
        Multiple files of unprocessed text but structured in xml also with sentence delimiter.
        Has some meta data of file and sentence id's for bilingual sentence pairing.

    - 'xml'
        Tokenized text in XML files with metadata about on document. Id's on tokens
        and sentences.

    - 'parsed'
        Similar to 'xml' but parsed through UDPipe.

    - 'freq'
        Frequency counts of words in corpura.
    """

    def __init__(self, corpuses: list = [], cache_dir: str = DEFAULT_CACHE_DIR, verbose: bool = False):
        assert(type(corpuses) is list), "parameter corpuses is of type {} - must be list".format(type(corpuses))
        self.cache_dir = cache_dir
        self.verbose = verbose
        for corp in corpuses:
            assert(corp in OPUS_MONO_DA['corpuses']), "{} not a valid corpus".format(corp)
        self.corpuses = corpuses

    def _download(self):
        for corp in self.corpuses:
            dataset_dir = download_dataset(corp,
                    process_func=_opus_process_func, cache_dir=self.cache_dir,
                    verbose = self.verbose)

    def _load(self):
        """Load corpuses into memory"""
        all_txt = []
        for corp in self.corpuses:
            with open(os.path.join(self.cache_dir,corp,"{}.txt".format(corp))) as txtfile:
                all_txt.append(txtfile.read())
        return "\n".join(all_txt)

    def load_as_txt(self):
        self._download()
        return self._load()

    def valid_corpuses(self):
        """
        Returns list of valid corpus elements to pass to the OPUS class
        when initializing.
        """
        return list(OPUS_MONO_DA['corpuses'].keys())

def _opus_process_func(tmp_file_path: str, meta_info: dict, cache_dir: str = DEFAULT_CACHE_DIR,
                          clean_up_raw_data: bool = True, verbose: bool = False):
    """
    Currently only processes .txt.gz files from the mono preprocessing format.
    """
    has_gz_compression = lambda url: re.match(r".*.gz$",url) != None
    destination = os.path.join(cache_dir, meta_info['name'], meta_info['name'] + meta_info['file_extension'])
    if has_gz_compression(meta_info['url']):
        import gzip
        with gzip.open(tmp_file_path, 'rb') as gfile:
            content = gfile.read()
            with open(destination,'wb') as txtfile:
                txtfile.write(content)
    else:
        raise Exception("Trying to download data with unknown compression \
        format via {}".format(meta_info['url']))
    os.remove(tmp_file_path)













