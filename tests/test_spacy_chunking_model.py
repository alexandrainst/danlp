import unittest

import spacy
import operator
from danlp.download import download_model, DEFAULT_CACHE_DIR, _unzip_process_func
from danlp.models import load_spacy_chunking_model
import os


class TestSpacyChunkingModel(unittest.TestCase):

    def test_predictions(self):

        chunker = load_spacy_chunking_model()
        some_text = "Jeg gik en tur med Lars"
        nlp = chunker.model
        doc = nlp(some_text)
        self.assertTrue(doc.is_tagged)
        self.assertTrue(doc.is_parsed)
        chunks = chunker.predict(some_text)
        self.assertEqual(len(chunks), len(doc))


if __name__ == '__main__':
    unittest.main()
