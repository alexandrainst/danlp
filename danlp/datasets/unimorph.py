import os
import pandas as pd
pd.options.mode.chained_assignment = None

from danlp.download import DEFAULT_CACHE_DIR, download_dataset, _unzip_process_func, DATASETS


class DaUnimorph():
    """
    Danish Unimorph.
    See also : https://unimorph.github.io/.

    The Danish Unimorph is a database which contains knowledge (lemmas and morphological features) 
    about different forms of nouns and verbs in Danish. 
    
    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):

        self.dataset_name = 'unimorph'
        self.file_extension = DATASETS[self.dataset_name]['file_extension']
        self.dataset_dir = download_dataset(self.dataset_name, process_func=_unzip_process_func, cache_dir=cache_dir)
        self.file_path = os.path.join(cache_dir, self.dataset_name + '.tsv')

        self.database = pd.read_csv(self.file_path, 
                                sep='\t', 
                                names=['lemma', 'form', 'feats'], 
                                encoding='unicode_escape', 
                                usecols=[0,1,2],
                                dtype={'lemma':str, 'form':str, 'feats':str})

        self.database['pos'] = self.database['feats'].apply(lambda feats: feats.split(';')[0])


    def load_with_pandas(self):
        """
        Loads the dataset in a dataframe

        :return: a dataframe

        """
        return self.database

    def get_inflections(self, form, pos=None, is_lemma = False, with_features=False):
        """
        Returns all possible inflections (forms) of a word (based on its lemma)

        :return: list of words

        """
        pos = _get_pos_list(pos)
        lemmas = [form] if is_lemma else self.get_lemmas(form, pos=pos)

        forms = []
        for p in pos:
            for lemma in lemmas:
                forms += self.database[(self.database['pos'] == p) & (self.database['lemma'] == lemma)].to_dict(orient='records')

        if with_features:
            return forms
        else:
            return [w['form'] for w in forms]

    def get_lemmas(self, form, pos=None, with_features=False):
        """
        Returns the lemma(s) of a word

        :return: list of lemmas

        """
        pos = _get_pos_list(pos)

        lemmas = []
        for p in pos:
            lemmas += self.database[(self.database['pos'] == p) & (self.database['form'] == form)].to_dict(orient='records')

        if with_features:
            return lemmas
        else:
            return [w['lemma'] for w in lemmas]


def _get_pos_list(pos):
    if pos == None:
        return ['N', 'V']
    elif type(pos) == str:
        return [pos]
    assert(type(pos) == list)
    return pos
