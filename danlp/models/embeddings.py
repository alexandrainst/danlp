import os

from gensim.models.keyedvectors import KeyedVectors

from danlp.download import MODELS, download_model, DEFAULT_CACHE_DIR

AVAILABLE_EMBEDDINGS = ['wiki.da.wv', 'cc.da.wv', 'connl.da.wv', 'news.da.wv']
AVAILABLE_SUBWORD_EMBEDDINGS = ['wiki.da.swv', 'cc.da.swv']


def load_wv_with_gensim(pretrained_embedding: str, cache_dir=DEFAULT_CACHE_DIR, verbose: bool = False):
    """

    Available wordembeddings:
    - wiki.da.wv
    - cc.da.wv
    - connl.da.wv
    - news.da.wv
    - wiki.da.swv
    - cc.da.swv

    :param pretrained_embedding:
    :param cache_dir: the directory for storing cached data
    :param verbose:
    :return: KeyedVectors or FastTextKeyedVectors
    """
    word_embeddings_available(pretrained_embedding, can_use_subword=True)  # TODO: Fix the subword thing for fasttext
    download_model(pretrained_embedding, cache_dir, _process_downloaded_embeddings, verbose=verbose)
    wv_path = os.path.join(cache_dir, pretrained_embedding + ".bin")

    if pretrained_embedding.split(".")[-1] == 'wv':
        return KeyedVectors.load_word2vec_format(wv_path, binary=True)

    elif pretrained_embedding.split(".")[-1] == 'swv':
        from gensim.models.fasttext import load_facebook_vectors
        return load_facebook_vectors(wv_path)


def load_wv_with_spacy(pretrained_embedding: str, cache_dir: str = DEFAULT_CACHE_DIR, verbose=False):
    """

    :param str pretrained_embedding:
    :param str cache_dir: the directory for storing cached data
    :param bool verbose:
    :return
    """
    import spacy
    word_embeddings_available(pretrained_embedding, can_use_subword=False)  # spaCy does not support subwords

    download_model(pretrained_embedding, cache_dir, _process_embeddings_for_spacy, verbose)

    spacy_model_dir = os.path.join(cache_dir, pretrained_embedding + ".spacy")

    return spacy.load(spacy_model_dir)


def load_keras_embedding_layer(pretrained_embedding: str, cache_dir=DEFAULT_CACHE_DIR, verbose=False, **kwargs):
    """

    :param pretrained_embedding:
    :param cache_dir: the directory for storing cached models
    :param verbose:
    :param kwargs: used to forward arguments to the keras Embedding layer
    :return:
    """
    word_embeddings_available(pretrained_embedding, can_use_subword=False)

    from keras.layers import Embedding
    wv = load_wv_with_gensim(pretrained_embedding, cache_dir, verbose)
    vocab_size = len(wv.vocab)

    if not 'trainable' in kwargs:
        kwargs['trainable'] = False

    embedding_layer = Embedding(vocab_size, wv.vector_size, weights=wv.vectors, **kwargs)

    return embedding_layer, wv.index2word


def load_pytorch_embedding_layer(pretrained_embedding: str, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """

    :param pretrained_embedding:
    :param cache_dir: the directory for storing cached models
    :return: an pytorch Embedding module and a list id2word
    """
    word_embeddings_available(pretrained_embedding, can_use_subword=False)
    import torch
    from torch.nn import Embedding

    word_vectors = load_wv_with_gensim(pretrained_embedding, cache_dir=cache_dir, verbose=verbose)
    weights = torch.FloatTensor(word_vectors.vectors)

    return Embedding.from_pretrained(weights), word_vectors.index2word


def load_context_embeddings_with_flair(direction = 'bi', word_embeddings = True, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """

    :param bidirectional:
    :param cache_dir:
    :param verbose:
    """
    from flair.embeddings import FlairEmbeddings
    from flair.embeddings import WordEmbeddings
    from flair.embeddings import StackedEmbeddings

    embeddings = []

    if word_embeddings:
        fasttext_embedding = WordEmbeddings('da')
        embeddings.append(fasttext_embedding)

    if direction == 'bi' or direction == 'fwd':
        fwd_weight_path = download_model('flair.fwd', cache_dir, verbose=verbose)
        embeddings.append(FlairEmbeddings(fwd_weight_path))

    if direction == 'bi' or direction == 'bwd':
        bwd_weight_path = download_model('flair.bwd', cache_dir, verbose=verbose)
        embeddings.append(FlairEmbeddings(bwd_weight_path))

    if len(embeddings) == 1:
        return embeddings[0]

    return StackedEmbeddings(embeddings=embeddings)


def word_embeddings_available(pretrained_embedding: str, can_use_subword=False):
    if not can_use_subword and pretrained_embedding in AVAILABLE_SUBWORD_EMBEDDINGS:
        raise ValueError("The framework does not support the use of subword pretrained embeddings")

    if pretrained_embedding not in AVAILABLE_EMBEDDINGS:
        if pretrained_embedding not in AVAILABLE_SUBWORD_EMBEDDINGS:
            raise ValueError("Pretrained embeddings {} do not exist".format(pretrained_embedding))


def _process_embeddings_for_spacy(embedding_file: str, cache_dir: str = '.data', verbose: bool = False):
    """
    To use pretrained embeddings with spaCy the embeddings need to be stored in
    a specific format. This function converts embeddings saved in the binary
    word2vec format to a spaCy model with the init_model() function from
    spaCy. The generated files will be saved in the cache_dir under a
    folder called <pretrained_embedding>.spacy

    More information on converting pretrained word embeddings to spaCy models here:
    https://spacy.io/usage/vectors-similarity#custom

    :param str embedding_file: the file name of the embedding binary file
    :param str cache_dir: the directory for storing cached data
    :param bool verbose:
    """
    from pathlib import Path
    from spacy.cli import init_model

    assert embedding_file[-4:] == '.bin'
    pretrained_embedding = embedding_file[:-4]

    vec_file = os.path.join(cache_dir, pretrained_embedding + ".vec")
    embedding_file_path = os.path.join(cache_dir, embedding_file)

    word_vecs = KeyedVectors.load_word2vec_format(embedding_file_path, binary=True, encoding='utf8')
    assert_wv_dimensions(word_vecs, pretrained_embedding)
    word_vecs.save_word2vec_format(vec_file, binary=False)

    spacy_dir = os.path.join(cache_dir, pretrained_embedding+'.spacy')
    os.makedirs(spacy_dir, exist_ok=True)

    if os.path.isabs(spacy_dir):
        full_spacy_dir = Path(spacy_dir)
    else:
        full_spacy_dir = Path(os.path.join(os.getcwd(), spacy_dir))

    init_model('da', full_spacy_dir, vectors_loc=vec_file)

    os.remove(vec_file)  # Clean up the vec file


def _process_downloaded_embeddings(tmp_file_path: str, clean_up_raw_data: bool = True, verbose: bool = False):
    """

    :param str tmp_file_path:
    :param bool clean_up_raw_data:
    :param bool verbose:
    """
    cache_dir = os.path.split(tmp_file_path)[0]
    tmp_filename = os.path.split(tmp_file_path)[1]
    pretrained_embedding = tmp_filename[:-4]

    bin_file_path = os.path.join(cache_dir, pretrained_embedding + ".bin")

    if pretrained_embedding in ['news.da.wv', 'wiki.da.wv']:
        if verbose:
            print("Converting {} embeddings to binary file format".format(pretrained_embedding))
        word_vecs = KeyedVectors.load_word2vec_format(tmp_file_path, binary=False, encoding='utf8')

        assert_wv_dimensions(word_vecs, pretrained_embedding)

        word_vecs.save_word2vec_format(bin_file_path, binary=True)

    elif pretrained_embedding == 'cc.da.wv':
        # Then it is a .gz file with a vec file inside
        import gzip
        import shutil

        vec_file_path = os.path.join(cache_dir, pretrained_embedding + ".vec")
        bin_file_path = os.path.join(cache_dir, pretrained_embedding + ".bin")
        if verbose:
            print("Decompressing raw {} embeddings".format(pretrained_embedding))

        with gzip.open(tmp_file_path, 'rb') as fin, open(vec_file_path, 'wb') as fout:
            shutil.copyfileobj(fin, fout)

        if verbose:
            print("Converting {} embeddings to binary file format".format(pretrained_embedding))
        word_vecs = KeyedVectors.load_word2vec_format(vec_file_path, binary=False, encoding='utf8')

        assert_wv_dimensions(word_vecs, pretrained_embedding)

        word_vecs.save_word2vec_format(bin_file_path, binary=True)

        os.remove(vec_file_path)  # Clean up the vec file

    elif pretrained_embedding == 'connl.da.wv':
        from zipfile import ZipFile
        if verbose:
            print("Unzipping raw {} embeddings".format(pretrained_embedding))

        with ZipFile(tmp_file_path, 'r') as zip_file:  # Extract files to cache_dir
            zip_file.extract("model.txt", path=cache_dir)

        org_vec_file = os.path.join(cache_dir, "model.txt")
        new_vec_file = os.path.join(cache_dir, "vecs.txt")
        ignored_lines = [138311, 260795, 550419, 638295, 727953, 851036, 865375, 878971, 1065332, 1135069, 1171719,
                         1331355, 1418396, 1463952, 1505510, 1587133]

        if verbose:
            print("Cleaning raw {} embeddings".format(pretrained_embedding))
        with open(org_vec_file, 'r', errors='replace') as fin, open(new_vec_file, 'w') as fout:
            for line_no, line in enumerate(fin, 1):
                if line_no == 1:
                    fout.write("1655870 100\n")
                elif line_no not in ignored_lines:
                    fout.write(line)
        if verbose:
            print("Converting {} embeddings to binary file format".format(pretrained_embedding))
        word_vecs = KeyedVectors.load_word2vec_format(new_vec_file, binary=False, encoding='utf8')

        assert_wv_dimensions(word_vecs, pretrained_embedding)

        word_vecs.save_word2vec_format(bin_file_path, binary=True)

        # Clean up the files
        os.remove(org_vec_file)
        os.remove(new_vec_file)

    elif pretrained_embedding == 'wiki.da.swv':
        from zipfile import ZipFile
        import random
        import string
        import shutil

        if verbose:
            print("Unzipping raw {} embeddings".format(pretrained_embedding))

        with ZipFile(tmp_file_path, 'r') as zip_file:  # Extract files to cache_dir

            tmp_path = os.path.join(cache_dir, ''.join(random.choice(string.ascii_lowercase) for i in range(6)))  # To not have name conflicts
            outpath = zip_file.extract("wiki.da.bin", path=tmp_path)

            os.rename(outpath, bin_file_path)
            shutil.rmtree(tmp_path)

    elif pretrained_embedding == 'cc.da.swv':
        import gzip
        import shutil

        bin_file_path = os.path.join(cache_dir, pretrained_embedding + ".bin")
        if verbose:
            print("Decompressing raw {} embeddings".format(pretrained_embedding))

        with gzip.open(tmp_file_path, 'rb') as fin, open(bin_file_path, 'wb') as fout:
            shutil.copyfileobj(fin, fout)

    else:
        raise NotImplementedError('There is not yet implemented any preprocessing for {}'.format(pretrained_embedding))

    if clean_up_raw_data:
        os.remove(tmp_file_path)


def assert_wv_dimensions(wv: KeyedVectors, pretrained_embedding: str):
    """
    This functions will check the dimensions of some wordembeddings wv,
    and check them against the data stored in WORD_EMBEDDINGS.

    :param gensim.models.KeyedVectors wv:
    :param str pretrained_embedding: the name of the pretrained embeddings
    """
    vocab_size = MODELS[pretrained_embedding]['vocab_size']
    embedding_dimensions = MODELS[pretrained_embedding]['dimensions']

    assert len(wv.vocab) == vocab_size, "Wrong vocabulary size, has the file been corrupted? Loaded vocab size: {}".format(len(wv.vocab))
    assert wv.vector_size == embedding_dimensions, "Wrong embedding dimensions, has the file been corrupted?"

