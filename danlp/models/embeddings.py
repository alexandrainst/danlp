"""
This module provides you with functions for loading 
pretrained Danish word embeddings through several NLP frameworks: 

    * flair
    * spaCy
    * Gensim

Available word embeddings:

    * wiki.da.wv
    * cc.da.wv
    * conll17.da.wv
    * news.da.wv
    * sketchengine.da.wv

Available subword embeddings:

    * wiki.da.swv
    * cc.da.swv
    * sketchengine.da.swv
"""

import os
from tempfile import TemporaryDirectory
from time import sleep

from gensim.models.keyedvectors import KeyedVectors

from danlp.download import MODELS, download_model, DEFAULT_CACHE_DIR, \
    _unzip_process_func

AVAILABLE_EMBEDDINGS = ['wiki.da.wv', 'cc.da.wv', 'conll17.da.wv',
                        'news.da.wv', 'sketchengine.da.wv', 'dslreddit.da.wv']
"""
"""

AVAILABLE_SUBWORD_EMBEDDINGS = ['wiki.da.swv', 'cc.da.swv',
                                'sketchengine.da.swv']
"""
"""


def load_wv_with_gensim(pretrained_embedding: str, cache_dir=DEFAULT_CACHE_DIR,
                        verbose: bool = False):
    """
    Loads word embeddings with Gensim.

    :param str pretrained_embedding:
    :param cache_dir: the directory for storing cached data
    :param bool verbose: `True` to increase verbosity
    :return: KeyedVectors or FastTextKeyedVectors
    """
    _word_embeddings_available(pretrained_embedding, can_use_subword=True)
    download_model(pretrained_embedding, cache_dir,
                   _process_downloaded_embeddings, verbose=verbose)
    wv_path = os.path.join(cache_dir, pretrained_embedding + ".bin")

    if pretrained_embedding.split(".")[-1] == 'wv':
        return KeyedVectors.load_word2vec_format(wv_path, binary=True)

    elif pretrained_embedding.split(".")[-1] == 'swv':
        from gensim.models.fasttext import load_facebook_vectors
        return load_facebook_vectors(wv_path)


def load_wv_with_spacy(pretrained_embedding: str,
                       cache_dir: str = DEFAULT_CACHE_DIR, verbose=False):
    """
    Loads a spaCy model with pretrained embeddings.

    :param str pretrained_embedding:
    :param str cache_dir: the directory for storing cached data
    :param bool verbose: `True` to increase verbosity
    :return: spaCy model
    """
    import spacy

    # spaCy does not support subwords
    _word_embeddings_available(pretrained_embedding, can_use_subword=False)

    spacy_model_dir = os.path.join(cache_dir, pretrained_embedding + ".spacy")

    if os.path.isdir(spacy_model_dir):
        # Return spaCy model if spaCy model dir exists
        return spacy.load(spacy_model_dir)

    bin_file_path = os.path.join(cache_dir, pretrained_embedding + ".bin")

    if os.path.isfile(bin_file_path):
        # Then we do not need to download the model
        model_info = MODELS[pretrained_embedding]
        model_info['name'] = pretrained_embedding
        _process_embeddings_for_spacy(bin_file_path[:-4] + ".tmp", model_info)
    else:
        download_model(pretrained_embedding, cache_dir,
                       _process_embeddings_for_spacy, verbose=True,
                       file_extension='.spacy')

    return spacy.load(spacy_model_dir)


def load_keras_embedding_layer(pretrained_embedding: str,
                               cache_dir=DEFAULT_CACHE_DIR, verbose=False,
                               **kwargs):
    """
    Loads a Keras Embedding layer.

    :param str pretrained_embedding:
    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    :param kwargs: used to forward arguments to the Keras Embedding layer
    :return: a Keras Embedding layer and index to word dictionary
    """
    _word_embeddings_available(pretrained_embedding, can_use_subword=False)

    from keras.layers import Embedding
    wv = load_wv_with_gensim(pretrained_embedding, cache_dir, verbose)
    vocab_size = len(wv.vocab)

    if not 'trainable' in kwargs:
        kwargs['trainable'] = False

    embedding_layer = Embedding(vocab_size, wv.vector_size, weights=wv.vectors,
                                **kwargs)

    return embedding_layer, wv.index2word


def load_pytorch_embedding_layer(pretrained_embedding: str,
                                 cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Loads a pytorch embbeding layer.

    :param str pretrained_embedding:
    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    :return: a pytorch Embedding module and a list id2word
    """
    _word_embeddings_available(pretrained_embedding, can_use_subword=False)
    import torch
    from torch.nn import Embedding

    word_vectors = load_wv_with_gensim(pretrained_embedding,
                                       cache_dir=cache_dir, verbose=verbose)
    weights = torch.FloatTensor(word_vectors.vectors)

    return Embedding.from_pretrained(weights), word_vectors.index2word


def load_context_embeddings_with_flair(direction='bi', word_embeddings=None,
                                       cache_dir=DEFAULT_CACHE_DIR,
                                       verbose=False):
    """
    Loads contextutal (dynamic) word embeddings with flair.

    :param str direction: bidirectional 'bi', forward 'fwd' or backward 'bwd'
    :param word_embedding:
    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    """
    from flair.embeddings import FlairEmbeddings
    from flair.embeddings import WordEmbeddings
    from flair.embeddings import StackedEmbeddings

    embeddings = []

    if word_embeddings is not None:
        _word_embeddings_available(word_embeddings, can_use_subword=False)
        download_model(word_embeddings, cache_dir,
                   _process_downloaded_embeddings, verbose=verbose)
        wv_path = os.path.join(cache_dir, word_embeddings + ".bin")
        
        fasttext_embedding = WordEmbeddings(wv_path)
        embeddings.append(fasttext_embedding)

    if direction == 'bi' or direction == 'fwd':
        fwd_weight_path = download_model('flair.fwd', cache_dir,
                                         verbose=verbose,
                                         process_func=_unzip_process_func)
        embeddings.append(FlairEmbeddings(fwd_weight_path))

    if direction == 'bi' or direction == 'bwd':
        bwd_weight_path = download_model('flair.bwd', cache_dir,
                                         verbose=verbose,
                                         process_func=_unzip_process_func)
        embeddings.append(FlairEmbeddings(bwd_weight_path))

    if len(embeddings) == 1:
        return embeddings[0]

    return StackedEmbeddings(embeddings=embeddings)


def _word_embeddings_available(pretrained_embedding: str,
                              can_use_subword=False):
    if not can_use_subword and pretrained_embedding in AVAILABLE_SUBWORD_EMBEDDINGS:
        raise ValueError(
            "The framework does not support the use of subword pretrained embeddings")

    if pretrained_embedding not in AVAILABLE_EMBEDDINGS:
        if pretrained_embedding not in AVAILABLE_SUBWORD_EMBEDDINGS:
            raise ValueError("Pretrained embeddings {} do not exist".format(
                pretrained_embedding))


def _process_embeddings_for_spacy(tmp_file_path: str, meta_info: dict,
                                  cache_dir: str = DEFAULT_CACHE_DIR,
                                  clean_up_raw_data: bool = True,
                                  verbose: bool = False):
    """
    To use pretrained embeddings with spaCy the embeddings need to be stored in
    a specific format. This function converts embeddings saved in the binary
    word2vec format to a spaCy model with the init_model() function from
    spaCy. The generated files will be saved in the cache_dir under a
    folder called <pretrained_embedding>.spacy

    More information on converting pretrained word embeddings to spaCy models here:
    https://spacy.io/usage/vectors-similarity#custom

    :param str tmp_file_path: the file name of the embedding binary file
    :param str cache_dir: the directory for storing cached data
    :param bool clean_up_raw_data: 
    :param bool verbose: `True` to increase verbosity
    """
    from pathlib import Path
    from spacy.cli import init_model

    embeddings = meta_info['name']

    bin_file_path = os.path.join(cache_dir, embeddings + ".bin")

    if not os.path.isfile(
            bin_file_path):  # Preprocess to transform to word2vec .bin format
        _process_downloaded_embeddings(tmp_file_path, meta_info, cache_dir,
                                       clean_up_raw_data, verbose)

    vec_file = embeddings + ".vec"

    word_vecs = KeyedVectors.load_word2vec_format(bin_file_path, binary=True,
                                                  encoding='utf8')
    assert_wv_dimensions(word_vecs, embeddings)
    word_vecs.save_word2vec_format(vec_file, binary=False)

    spacy_dir = os.path.join(cache_dir, embeddings + '.spacy')
    os.makedirs(spacy_dir, exist_ok=True)

    if os.path.isabs(spacy_dir):
        full_spacy_dir = Path(spacy_dir)
    else:
        full_spacy_dir = Path(os.path.join(os.getcwd(), spacy_dir))

    init_model('da', full_spacy_dir, vectors_loc=vec_file)

    os.remove(vec_file)  # Clean up the vec file


def _process_downloaded_embeddings(tmp_file_path: str, meta_info: dict,
                                   cache_dir: str = DEFAULT_CACHE_DIR,
                                   clean_up_raw_data: bool = True,
                                   verbose: bool = False):
    """

    :param str tmp_file_path:
    :param dict meta_info:
    :param str cache_dir: the directory for storing cached data
    :param bool clean_up_raw_data:
    :param bool verbose: `True` to increase verbosity
    """
    pretrained_embedding = meta_info['name']

    bin_file_path = os.path.join(cache_dir, pretrained_embedding + ".bin")

    if pretrained_embedding in ['news.da.wv', 'wiki.da.wv']:
        if verbose:
            print("Converting {} embeddings to binary file format".format(
                pretrained_embedding))
        word_vecs = KeyedVectors.load_word2vec_format(tmp_file_path,
                                                      binary=False,
                                                      encoding='utf8')

        assert_wv_dimensions(word_vecs, pretrained_embedding)

        word_vecs.save_word2vec_format(bin_file_path, binary=True)

    elif pretrained_embedding == 'sketchengine.da.wv':
        new_vec_file = os.path.join(cache_dir, "vecs.txt")

        if verbose:
            print("Cleaning raw {} embeddings".format(pretrained_embedding))
        with open(tmp_file_path, 'r', errors='replace') as fin, open(new_vec_file, 'w') as fout:
            for line_no, line in enumerate(fin, 1):
                if line_no == 1:
                    fout.write("2360830 100\n")
                elif len(line.split()) <= 101:
                    fout.write(line)

        word_vecs = KeyedVectors.load_word2vec_format(new_vec_file, binary=False, encoding='utf8')

        assert_wv_dimensions(word_vecs, pretrained_embedding)

        word_vecs.save_word2vec_format(bin_file_path, binary=True)

        os.remove(new_vec_file)  # Clean up the vec file


    elif pretrained_embedding == 'cc.da.wv':
        # Then it is a .gz file with a vec file inside
        import gzip
        import shutil

        vec_file_path = os.path.join(cache_dir, pretrained_embedding + ".vec")
        bin_file_path = os.path.join(cache_dir, pretrained_embedding + ".bin")
        if verbose:
            print(
                "Decompressing raw {} embeddings".format(pretrained_embedding))

        with gzip.open(tmp_file_path, 'rb') as fin, open(vec_file_path,
                                                         'wb') as fout:
            shutil.copyfileobj(fin, fout)

        if verbose:
            print("Converting {} embeddings to binary file format".format(
                pretrained_embedding))
        word_vecs = KeyedVectors.load_word2vec_format(vec_file_path,
                                                      binary=False,
                                                      encoding='utf8')

        assert_wv_dimensions(word_vecs, pretrained_embedding)

        word_vecs.save_word2vec_format(bin_file_path, binary=True)

        os.remove(vec_file_path)  # Clean up the vec file

    elif pretrained_embedding == 'conll17.da.wv':
        from zipfile import ZipFile
        if verbose:
            print("Unzipping raw {} embeddings".format(pretrained_embedding))

        with ZipFile(tmp_file_path,
                     'r') as zip_file:  # Extract files to cache_dir
            zip_file.extract("model.txt", path=cache_dir)

        org_vec_file = os.path.join(cache_dir, "model.txt")
        new_vec_file = os.path.join(cache_dir, "vecs.txt")
        ignored_lines = [138311, 260795, 550419, 638295, 727953, 851036,
                         865375, 878971, 1065332, 1135069, 1171719,
                         1331355, 1418396, 1463952, 1505510, 1587133]

        if verbose:
            print("Cleaning raw {} embeddings".format(pretrained_embedding))
        with open(org_vec_file, 'r', errors='replace') as fin, open(
                new_vec_file, 'w') as fout:
            for line_no, line in enumerate(fin, 1):
                if line_no == 1:
                    fout.write("1655870 100\n")
                elif line_no not in ignored_lines:
                    fout.write(line)
        if verbose:
            print("Converting {} embeddings to binary file format".format(
                pretrained_embedding))
        word_vecs = KeyedVectors.load_word2vec_format(new_vec_file,
                                                      binary=False,
                                                      encoding='utf8')

        assert_wv_dimensions(word_vecs, pretrained_embedding)

        word_vecs.save_word2vec_format(bin_file_path, binary=True)

        # Clean up the files
        os.remove(org_vec_file)
        os.remove(new_vec_file)

    elif pretrained_embedding == 'dslreddit.da.wv':
        _process_dslreddit(tmp_file_path, cache_dir)

    elif pretrained_embedding == 'wiki.da.swv':
        _unzip_process_func(tmp_file_path, clean_up_raw_data, verbose,
                            file_in_zip='wiki.da.bin')

    elif pretrained_embedding == 'cc.da.swv':
        import gzip
        import shutil

        bin_file_path = os.path.join(cache_dir, pretrained_embedding + ".bin")
        if verbose:
            print(
                "Decompressing raw {} embeddings".format(pretrained_embedding))

        with gzip.open(tmp_file_path, 'rb') as fin, open(bin_file_path,
                                                         'wb') as fout:
            shutil.copyfileobj(fin, fout)

    elif pretrained_embedding == 'sketchengine.da.swv':
        import shutil
        bin_file_path = os.path.join(cache_dir, pretrained_embedding + ".bin")

        shutil.copy(tmp_file_path, bin_file_path)

    else:
        raise NotImplementedError(
            'There is not yet implemented any preprocessing for {}'.format(
                pretrained_embedding))

    if clean_up_raw_data:
        os.remove(tmp_file_path)


def _process_dslreddit(tmp_file_path: str, cache_dir: str,
                       embedding_name: str = "dslreddit.da.wv"):
    from zipfile import ZipFile

    tmp_dir = TemporaryDirectory()
    with ZipFile(tmp_file_path, 'r') as zip_file:  # Extract files to cache_dir
        zip_file.extractall(path=tmp_dir.name)

    tmp_wv_path = os.path.join(tmp_dir.name, "word2vec_dsl_sentences_reddit_sentences_300_cbow_negative.kv")

    word_vecs = KeyedVectors.load(tmp_wv_path, mmap='r')
    assert_wv_dimensions(word_vecs, embedding_name)

    bin_file_path = os.path.join(cache_dir, embedding_name + ".bin")

    word_vecs.save_word2vec_format(bin_file_path, binary=True)
    tmp_dir.cleanup()


def assert_wv_dimensions(wv: KeyedVectors, pretrained_embedding: str):
    """
    This function will check the dimensions of some word embeddings wv,
    and check them against the data stored in WORD_EMBEDDINGS.

    :param gensim.models.KeyedVectors wv: word embeddings
    :param str pretrained_embedding: the name of the pretrained embeddings
    """
    vocab_size = MODELS[pretrained_embedding]['vocab_size']
    embedding_dimensions = MODELS[pretrained_embedding]['dimensions']

    vocab_err_msg = "Wrong vocabulary size, has the file been corrupted? " \
                "Loaded vocab size: {}".format(len(wv.vocab))

    dim_err_msg = "Wrong embedding dimensions, has the file been corrupted?"

    assert len(wv.vocab) == vocab_size, vocab_err_msg
    assert wv.vector_size == embedding_dimensions, dim_err_msg

if __name__ == '__main__':
    _process_dslreddit('/home/alexandra/Downloads/word2vec_dsl_sentences_reddit_sentences_300_cbow_negative.zip', '/home/alexandra/.danlp')