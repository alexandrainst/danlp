import os
import random

from danlp.download import download_dataset, DEFAULT_CACHE_DIR, DATASETS


class WikiAnn:
    """
    Class for loading the WikiANN dataset.

    :param str cache_dir: the directory for storing cached models

    """
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.dataset_name = 'wikiann'
        self.file_extension = DATASETS[self.dataset_name]['file_extension']

        self.dataset_dir = download_dataset(self.dataset_name, process_func=_wikiann_process_func, cache_dir=cache_dir)

    def load_with_flair(self, predefined_splits: bool = False):
        """
        Loads the dataset with flair.

        :param bool predefined_splits:
        :return: ColumnCorpus
        """
        from flair.data import Corpus
        from flair.datasets import ColumnCorpus

        columns = {0: 'text', 3: 'ner'}

        # init a corpus using column format, data folder and the names of the train, dev and test files
        corpus: Corpus = ColumnCorpus(self.dataset_dir, columns, train_file=self.dataset_name + self.file_extension)
        return corpus

    def load_with_spacy(self):
        """
        Loads the dataset with spaCy. 

        This function will convert the CoNLL02/03 format to json format for spaCy.
        As the function will return a spacy.gold.GoldCorpus which needs a dev set
        this function also splits the dataset into a 70/30 split as is done by
        Pan et al. (2017).

        - Pan et al. (2017): https://aclweb.org/anthology/P17-1178
        
        :return: GoldCorpus
        """
        import srsly
        from spacy.cli.converters import conll_ner2json
        from spacy.gold import GoldCorpus
        from spacy.gold import Path

        conll_path = os.path.join(self.dataset_dir, self.dataset_name + self.file_extension)
        dev_json_path = os.path.join(self.dataset_dir, self.dataset_name + "dev.json")
        train_json_path = os.path.join(self.dataset_dir, self.dataset_name + "train.json")

        if not os.path.isfile(dev_json_path) or not os.path.isfile(train_json_path):
            # Convert the conll ner files to json
            with open(conll_path, 'r') as file:
                file_as_string = file.read()
                # n_sents=0 means we do not group the sentences into documents
                file_as_json = conll_ner2json(file_as_string, n_sents=0,
                                              no_print=True)

                all_sents = file_as_json[0]['paragraphs'][0]['sentences']

                random.seed(42)
                random.shuffle(all_sents)

                train_size = round(len(all_sents) * 0.7)
                train_sents = all_sents[:train_size]
                dev_sents = all_sents[train_size:]

                train_json = [{'id': 0, 'paragraphs': [{'sentences': train_sents}]}]
                dev_json = [{'id': 0, 'paragraphs': [{'sentences': dev_sents}]}]

                srsly.write_json(train_json_path, train_json)
                srsly.write_json(dev_json_path, dev_json)

        assert os.path.isfile(train_json_path) and os.path.isfile(train_json_path)

        return GoldCorpus(Path(train_json_path), Path(dev_json_path))


def _convert_wikiann_to_iob(org_file, dest_file):
    """
    Converts the original WikiANN format to a CoNLL02/03 format.
    However as the WikiANN dataset do not contain any marking of
    what sentences belongs to a document we omit using the line

    -DOCSTART- -X- O O

    As used in the CoNLL02/03 format. This format is known in
    spaCy as the CoNLL 2003 NER format.

    :param org_file:
    :param dest_file:
    """
    with open(org_file, 'r', encoding='utf-8') as file:
        sentence_counter = 0
        sentences = []

        sent_tokens = []
        sent_tags = []
        for line in file:
            if line == "\n":  # End of sentence
                assert len(sent_tokens) == len(sent_tags)

                sentences.append({'tokens': sent_tokens, 'tags': sent_tags})
                sent_tokens, sent_tags = [], []
                sentence_counter += 1
                continue

            columns = line.split(" ")
            token = columns[0]
            tag = columns[-1].replace("\n", "")

            assert len(token) > 0
            assert (tag == 'O') or (tag.split("-")[1] in ['ORG', 'PER', 'LOC'])

            if u"\xa0" in token:
                # Some of the tokens contains a 'no-break space' char (0xA0).
                # They should be separate tokens.
                extra_tokens = token.split(u"\xa0")
                sent_tokens.extend(extra_tokens)

                tag_value = tag.split("-")[1]
                tags = ["I-" + tag_value for i in extra_tokens]

                if tag.split("-")[0] == 'B':
                    tags[0] = "I-" + tag_value
                sent_tags.extend(tags)

            else:
                sent_tokens.append(token)
                sent_tags.append(tag)

        # Write to new file
        with open(dest_file, 'w') as destination_file:
            for sent in sentences:
                tok_lines = zip(sent['tokens'], sent['tags'])
                lines = ["{} _ _ {}".format(tok, tag) for i, (tok, tag) in enumerate(tok_lines)]
                destination_file.write("\n".join(lines) + "\n\n")


def _wikiann_process_func(tmp_file_path: str, meta_info: dict, cache_dir: str = DEFAULT_CACHE_DIR,
                          clean_up_raw_data: bool = True, verbose: bool = False):
    import tarfile
    destination = os.path.join(cache_dir, meta_info['name'], meta_info['name'] + meta_info['file_extension'])

    with tarfile.open(tmp_file_path) as file:
        file.extract('wikiann-da.bio', '/tmp/')

    _convert_wikiann_to_iob('/tmp/wikiann-da.bio', destination)

    os.remove(tmp_file_path)
    os.remove('/tmp/wikiann-da.bio')
