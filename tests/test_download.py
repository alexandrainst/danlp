import os
import unittest

from danlp.download import MODELS, download_model


class TestDownload(unittest.TestCase):

    def test_all_downloadable_files_has_checksums(self):
        for model, data in MODELS.items():
            self.assertIn('size', data, msg="Model {}".format(model))
            self.assertIn('md5_checksum', data)
            self.assertIn('file_extension', data)

    def test_download_fails_with_wrong_embedding_title(self):
        with self.assertRaises(ValueError):
            download_model('do.not.exists.wv')


if __name__ == '__main__':
    unittest.main()
