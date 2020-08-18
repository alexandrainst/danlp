from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func


def load_spacy_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False, textcat=None, vectorError=False):
    """
    Loads a spacy model.
    
    OBS vectorError is a TEMP ugly work around error encounted by keeping two models an not been able to find referece name for vectros
    """
    from spacy.util import load_model_from_path

    if textcat==None or vectorError==True:
        modelname='spacy'

        model_weight_path = download_model(modelname, cache_dir,
                                           process_func=_unzip_process_func,
                                           verbose=verbose)
        nlp = load_model_from_path(model_weight_path)
        
    
    if textcat=='sentiment':
        modelname='spacy.sentiment'
        
        model_weight_path = download_model(modelname, cache_dir,
                                           process_func=_unzip_process_func,
                                           verbose=verbose)
        # quick fix from not aligned models storage:
        import os
        model_weight_path =  os.path.join(model_weight_path, 'spacy.sentiment')
        
        nlp = load_model_from_path(model_weight_path)
    
    
    return nlp