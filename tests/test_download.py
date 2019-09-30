import unittest
from typing import Callable

from danlp.datasets.wiki_ann import _wikiann_process_func
from danlp.datasets.word_sim import _word_sim_process_func
from danlp.download import MODELS, download_model, DATASETS, download_dataset, _unzip_process_func, _check_process_func
from danlp.models.embeddings import _process_downloaded_embeddings, _process_embeddings_for_spacy


class TestDownload(unittest.TestCase):

    def test_all_downloadable_files_has_checksums(self):
        for model, data in MODELS.items():
            self.assertIn('size', data, msg="Model {}".format(model))
            self.assertIn('md5_checksum', data)
            self.assertIn('file_extension', data)

        for dataset, data in DATASETS.items():
            self.assertIn('size', data, msg="Dataset {}".format(dataset))
            self.assertIn('md5_checksum', data)
            self.assertIn('file_extension', data)

    def test_download_fails_with_wrong_title(self):
        with self.assertRaises(ValueError):
            download_model('do.not.exists.wv')

        with self.assertRaises(ValueError):
            download_dataset('do.not.exists.zip')

    def test_process_functions(self):
        process_functions = [
            _process_downloaded_embeddings,
            _process_embeddings_for_spacy,
            _unzip_process_func,
            _wikiann_process_func,
            _word_sim_process_func
        ]

        for proc_func in process_functions:
            self.assertIsInstance(proc_func, Callable)
            try:
                _check_process_func(proc_func)
            except AssertionError:
                self.fail("{} does not have the correct arguments".format(proc_func))


if __name__ == '__main__':
    unittest.main()
