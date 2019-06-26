from danlp.download import download_model, DEFAULT_CACHE_DIR


def load_pos_tagger_with_flair(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """

    :param cache_dir:
    :param verbose:
    :return:
    """
    from flair.models import SequenceTagger

    model_weight_path = download_model('flair.pos', cache_dir, verbose=verbose)

    # using the flair model
    flair_model = SequenceTagger.load_from_file(model_weight_path)

    return flair_model
