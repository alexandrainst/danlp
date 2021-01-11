import os
import pandas as pd
import json

from danlp.download import DEFAULT_CACHE_DIR, download_dataset, _unzip_process_func, DATASETS


class DanNet():
    """
    DanNet wrapper, providing functions to access the main features of DanNet.
    See also : https://cst.ku.dk/projekter/dannet/.

    Dannet consists of a set of 4 databases: 

        * words
        * word senses
        * relations
        * synsets

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity

    """

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):

        self.dataset_name = 'dannet'
        self.file_extension = DATASETS[self.dataset_name]['file_extension']

        self.dataset_dir = download_dataset(self.dataset_name, process_func=_unzip_process_func, cache_dir=cache_dir)

        self.words = pd.read_csv(os.path.join(self.dataset_dir, "words.csv"), 
                                sep='@', 
                                names=['word_id', 'form', 'pos', 'nan'], 
                                encoding='unicode_escape', 
                                usecols=[0,1,2], 
                                dtype={'word_id':str})
        self.wordsenses = pd.read_csv(os.path.join(self.dataset_dir, "wordsenses.csv"), 
                                    sep='@', 
                                    names=['wordsense_id', 'word_id', 'synset_id', 'register', 'nan'], 
                                    encoding='unicode_escape', 
                                    usecols=[1,2],
                                    dtype={'wordsense_id':str, 'word_id':str, 'synset_id':str})
        self.relations = pd.read_csv(os.path.join(self.dataset_dir, "relations.csv"), 
                                    sep='@', 
                                    names=['synset_id', 'wordnetowl', 'relation', 'value', 'taxonomic', 'inheritance_comment', 'nan'], 
                                    encoding='unicode_escape', 
                                    usecols=[0,1,2,3,4,5],
                                    dtype={'synset_id':str, 'value':str})
        self.synsets = pd.read_csv(os.path.join(self.dataset_dir, "synsets.csv"), 
                                    sep='@', 
                                    names=['synset_id', 'label', 'gloss', 'ontological_type'], 
                                    encoding='unicode_escape', 
                                    usecols=[0,1,2,3],
                                    dtype={'synset_id':str})

    def load_with_pandas(self):
        """
        Loads the datasets in 4 dataframes

        :return: 4 dataframes: words, wordsenses, relations, synsets

        """
        return self.words, self.wordsenses, self.relations, self.synsets


    def synonyms(self, word, pos=None):
        """
        Returns the synonyms of `word`. 

        :param word: text
        :param pos: (list of) part of speech tag(s) (in "Noun", "Verb", "Adjective")
        :return: list of synonyms

        :Example:

            "`hav`" 
            returns
            ["sø", "ocean"]
        """

        word_ids = self._word_ids(word, pos)
        synset_ids = self._synset_ids(word, pos)
        synonym_ids = self.wordsenses[self.wordsenses['synset_id'].isin(synset_ids) & ~self.wordsenses['word_id'].isin(word_ids)]['word_id'].tolist()
        synonyms = self.words[self.words['word_id'].isin(synonym_ids)]['form'].tolist()
        return synonyms

    def meanings(self, word, pos=None):
        """
        Returns the meanings of `word`.

        :param word: text
        :param pos: (list of) part of speech tag(s) (in "Noun", "Verb", "Adjective")
        :return: list of meanings

        """
        
        synset_ids = self._synset_ids(word, pos)
        meanings = self.synsets[self.synsets['synset_id'].isin(synset_ids)]['gloss'].tolist()
        
        return meanings


    def hypernyms(self, word, pos=None):
        """
        Returns the hypernyms of `word`.

        :param word: text
        :param pos: (list of) part of speech tag(s) (in "Noun", "Verb", "Adjective")
        :return: list of hypernyms

        """
        
        word_synset_ids = self._synset_ids(word)
        hyper_synset_ids = self.relations[self.relations['synset_id'].isin(word_synset_ids) & (self.relations['relation']=='has_hyperonym')]['value'].tolist()
        hyper_synset_ids += self.relations[self.relations['value'].isin(word_synset_ids) & (self.relations['relation']=='has_hyponym')]['synset_id'].tolist()
        hyper_synset_ids = [val for val in hyper_synset_ids if val.isdigit()]
        hypernyms_ids = self.wordsenses[self.wordsenses['synset_id'].isin(hyper_synset_ids)]['word_id'].tolist()
        hypernyms = self.words[self.words['word_id'].isin(hypernyms_ids)]['form'].tolist()
        
        return hypernyms


    def hyponyms(self, word, pos=None):
        """
        Returns the hyponyms of `word`.

        :param word: text
        :param pos: (list of) part of speech tag(s) (in "Noun", "Verb", "Adjective")
        :return: list of hypernyms

        """
        
        word_synset_ids = self._synset_ids(word, pos)
        hypo_synset_ids = self.relations[self.relations['synset_id'].isin(word_synset_ids) & (self.relations['relation']=='has_hyponym')]['value'].tolist()
        hypo_synset_ids += self.relations[self.relations['value'].isin(word_synset_ids) & (self.relations['relation']=='has_hyperonym')]['synset_id'].tolist()
        hypo_synset_ids = [val for val in hypo_synset_ids if val.isdigit()]
        hypernyms_ids = self.wordsenses[self.wordsenses['synset_id'].isin(hypo_synset_ids)]['word_id'].tolist()
        hypernyms = self.words[self.words['word_id'].isin(hypernyms_ids)]['form'].tolist()
        
        return hypernyms

    def domains(self, word, pos=None):
        """
        Returns the domains of `word`.

        :param word: text
        :param pos: (list of) part of speech tag(s) (in "Noun", "Verb", "Adjective")
        :return: list of domains

        """
        
        word_synset_ids = self._synset_ids(word, pos)
        dom_synset_ids = self.relations[self.relations['synset_id'].isin(word_synset_ids) & (self.relations['relation']=='domain')]['value'].tolist()
        dom_synset_ids = [val for val in dom_synset_ids if val.isdigit()]
        domains_ids = self.wordsenses[self.wordsenses['synset_id'].isin(dom_synset_ids)]['word_id'].tolist()
        domains = self.words[self.words['word_id'].isin(domains_ids)]['form'].tolist()
        
        return domains

    def wordnet_relations(self, word, pos=None, eurowordnet=True):
        """
        Returns the name of the relations `word` is associated with.

        :param word: text
        :param pos: (list of) part of speech tag(s) (in "Noun", "Verb", "Adjective")
        :return: list of relations 

        """
        if eurowordnet:
            rel_name = "relation"
        else:
            rel_name = "wordnetowl"
        
        synset_ids = self._synset_ids(word, pos)
        relations = self.relations[self.relations['synset_id'].isin(synset_ids)][rel_name].tolist()
        
        return set(relations)



    def pos(self, word):
        """
        Returns the part-of-speech tags `word` can be categorized with among "Noun", "Verb" or "Adjective".

        :param word: text
        :return: list of part-of-speech tags
        """

        return list(self.words[self.words['form'] == word]['pos'].unique())

    def _word_ids(self, word, pos=None):

        pos = _get_pos_list(pos)
        word = word.lower()

        return self.words[(self.words['form'] == word) & self.words['pos'].isin(pos)]['word_id'].tolist()

    def _synset_ids(self, word, pos=None):

        word_ids = self._word_ids(word, pos)
        return self.wordsenses[self.wordsenses['word_id'].isin(word_ids)]['synset_id'].tolist()

    def _word_from_id(self, word_id):

        assert(type(word_id) == int or (type(word_id) == str and word_id.isdigit()))
        word_id = str(word_id)

        return self.words[self.words['word_id'] == word_id]['form'].tolist()

    def _synset_from_id(self, synset_id):

        assert(type(synset_id) == int or (type(synset_id) == str and synset_id.isdigit()))
        synset_id = str(synset_id)

        synset_labels = self.synsets[self.synsets['synset_id'] == synset_id]['label'].tolist()
        return set([w.split('_')[0] for s in synset_labels for w in s[1:-1].split('; ')])


    def __str__(self):

        return "DanNet: {} word forms, {} lexemes, {} synsets".format(len(set(self.words['form'])), len(self.words['word_id']), len(set(self.wordsenses['synset_id'])))


def _get_pos_list(pos):
    if pos == None:
        return ['Noun', 'Verb', 'Adjective']
    elif type(pos) == str:
        return [pos]
    assert(type(pos) == list)
    return pos

