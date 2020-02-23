from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func


def load_spacy_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Loads a spacy model.
    """
    from spacy.util import load_model_from_path

    model_weight_path = download_model('spacy', cache_dir,
                                       process_func=_unzip_process_func,
                                       verbose=verbose)

    nlp = load_model_from_path(model_weight_path)

    return nlp
