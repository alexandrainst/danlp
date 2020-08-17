from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func


def load_spacy_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False, textcat=None):
    """
    Loads a spacy model.
    """
    from spacy.util import load_model_from_path

    modelname='spacy'
    
    
    model_weight_path = download_model(modelname, cache_dir,
                                       process_func=_unzip_process_func,
                                       verbose=verbose)
    
    nlp = load_model_from_path(model_weight_path)
    
    
    # OBS temparary ugly fix to not get da.vecotrs not found is to load the original danlp model before the sentiment model 
    if textcat=='sentiment':
        import os
        modelname='spacy.sentiment'
        
        model_weight_path = download_model(modelname, cache_dir,
                                       process_func=_unzip_process_func,
                                       verbose=verbose)
        # quick fix from not aligned models storage
        model_weight_path =  os.path.join(model_weight_path, 'spacy.sentiment')
    
        nlp = load_model_from_path(model_weight_path)
    

    return nlp
