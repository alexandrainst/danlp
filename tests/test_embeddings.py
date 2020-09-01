import unittest

from gensim.models.keyedvectors import FastTextKeyedVectors

from danlp.download import MODELS, download_model, _unzip_process_func
from danlp.models.embeddings import load_wv_with_spacy, load_wv_with_gensim, load_context_embeddings_with_flair, \
    AVAILABLE_EMBEDDINGS, AVAILABLE_SUBWORD_EMBEDDINGS


class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        # First we will add smaller test embeddings to the
        MODELS['wiki.da.small.wv'] = {
            'url': 'https://danlp.alexandra.dk/304bd159d5de/tests/wiki.da.small.zip',
            'vocab_size': 5000,
            'dimensions': 300,
            'md5_checksum': 'fcaa981a613b325ae4dc61aba235aa82',
            'size': 5594508,
            'file_extension': '.bin'
        }

        AVAILABLE_EMBEDDINGS.append('wiki.da.small.wv')

        self.embeddings_for_testing = [
            'wiki.da.small.wv',
            'dslreddit.da.wv'
        ]
        # Lets download the models and unzip it
        for emb in self.embeddings_for_testing:
            download_model(emb, process_func=_unzip_process_func)

    def test_embeddings_with_spacy(self):
        with self.assertRaises(ValueError):
            load_wv_with_spacy("wiki.da.small.swv")

        embeddings = load_wv_with_spacy("wiki.da.wv")

        sentence = embeddings('jeg gik ned af en gade')
        for token in sentence:
            self.assertTrue(token.has_vector)

    def test_embeddings_with_gensim(self):
        for emb in self.embeddings_for_testing:
            embeddings = load_wv_with_gensim(emb)
            self.assertEqual(MODELS[emb]['vocab_size'], len(embeddings.vocab))


    def test_embeddings_with_flair(self):
        from flair.data import Sentence

        embs = load_context_embeddings_with_flair(word_embeddings='wiki.da.wv')

        sentence1 = Sentence('Han fik bank')
        sentence2 = Sentence('Han fik en ny bank')

        embs.embed(sentence1)
        embs.embed(sentence2)

        # Check length of context embeddings
        self.assertEqual(len(sentence1[2].embedding), 2364)
        self.assertEqual(len(sentence2[4].embedding), 2364)

    def test_fasttext_embeddings(self):
        # First we will add smaller test embeddings to the
        MODELS['ddt.swv'] = {
            'url': 'https://danlp.alexandra.dk/304bd159d5de/tests/ddt.swv.zip',
            'vocab_size': 5000,
            'dimensions': 100,
            'md5_checksum': 'c50c61e1b434908e2732c80660abf8bf',
            'size': 741125088,
            'file_extension': '.bin'
        }

        AVAILABLE_SUBWORD_EMBEDDINGS.append('ddt.swv')

        download_model('ddt.swv', process_func=_unzip_process_func)

        fasttext_embeddings = load_wv_with_gensim('ddt.swv')

        self.assertEqual(type(fasttext_embeddings), FastTextKeyedVectors)

        # The word is not in the vocab
        self.assertNotIn('institutmedarbejdskontrakt', fasttext_embeddings.vocab)

        # However we can get an embedding because of subword units
        self.assertEqual(fasttext_embeddings['institutmedarbejdskontrakt'].size, 100)


if __name__ == '__main__':
    unittest.main()
