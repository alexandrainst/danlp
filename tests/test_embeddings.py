import unittest

from gensim.models.keyedvectors import FastTextKeyedVectors

from danlp.models.embeddings import load_wv_with_spacy, load_wv_with_gensim, load_context_embeddings_with_flair


class TestEmbeddings(unittest.TestCase):

    def test_embeddings_with_spacy(self):
        with self.assertRaises(ValueError):
            load_wv_with_spacy("wiki.da.swv")

        embeddings = load_wv_with_spacy("wiki.da.wv")

        sentence = embeddings('jeg gik ned af en gade')
        for token in sentence:
            self.assertTrue(token.has_vector)

    def test_embeddings_with_gensim(self):
        embeddings = load_wv_with_gensim('connl.da.wv')

        most_similar = embeddings.most_similar(positive=['københavn', 'england'], negative=['danmark'], topn=1)

        self.assertEqual(most_similar[0], ('london', 0.7156291604042053))

    def test_embeddings_with_flair(self):
        from flair.data import Sentence

        embs = load_context_embeddings_with_flair()

        sentence1 = Sentence('Han fik bank')
        sentence2 = Sentence('Han fik en ny bank')

        embs.embed(sentence1)
        embs.embed(sentence2)

        # Check length of context embeddings
        self.assertEqual(len(sentence1[2].embedding), 2364)
        self.assertEqual(len(sentence2[4].embedding), 2364)

        # Show the embeddings are different
        self.assertEqual(int(sum(sentence2[4].embedding == sentence1[2].embedding)), 52)


    def test_fasttext_embeddings(self):
        fasttext_embeddings = load_wv_with_gensim('wiki.da.swv')

        self.assertEqual(type(fasttext_embeddings), FastTextKeyedVectors)

        # The word is not in the vocab
        self.assertNotIn('institutmedarbejdskontrakt', fasttext_embeddings.vocab)

        # However we can get an embedding because of subword units
        self.assertEqual(fasttext_embeddings['institutmedarbejdskontrakt'].size, 300)

if __name__ == '__main__':
    unittest.main()