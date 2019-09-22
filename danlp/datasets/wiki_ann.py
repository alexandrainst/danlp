import os
import string

from danlp.download import download_dataset, DEFAULT_CACHE_DIR, DATASETS
from danlp.utils import random_string


class WikiAnn:

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.dataset_name = 'wikiann'
        self.file_extension = DATASETS[self.dataset_name]['file_extension']

        self.dataset_dir = download_dataset(self.dataset_name, process_func=_wikiann_process_func, cache_dir=cache_dir)

    def load_ner_with_flair(self, predefined_splits: bool = False):

        from flair.data import Corpus
        from flair.datasets import ColumnCorpus

        columns = {1: 'text', 2: 'ner'}

        # init a corpus using column format, data folder and the names of the train, dev and test files
        corpus: Corpus = ColumnCorpus(self.dataset_dir, columns,
                                      train_file=self.dataset_name+self.file_extension)
        return corpus


def _convert_wikiann_to_iob(org_file, dest_file):
    """
    # sent_id = 0
    # text = Peter sagde hej
    1	Peter   B-PER
    2   sagde   O
    3   hej 0
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
                reassembly = ' '.join(sent_tokens).replace(' , ', ',').replace(' .', '.').replace(' !', '!')
                reassembly = reassembly.replace(' ?', '?').replace(' : ', ': ').replace(' \'', '\'')

                sentences.append({
                    'id': sentence_counter,
                    'text': reassembly,
                    'tokens': sent_tokens,
                    'tags': sent_tags
                })

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
                string = "# sent_id = {}\n#text = {}\n".format(sent['id'], sent['text'])
                tok_lines = zip(sent['tokens'], sent['tags'])
                lines = [ "{}\t{}\t{}".format(i,tok,tag) for i, (tok, tag) in enumerate(tok_lines)]
                string += "\n".join(lines)
                string += "\n\n"

                destination_file.write(string)


def _wikiann_process_func(tmp_file_path: str, meta_info: dict, cache_dir: str = DEFAULT_CACHE_DIR,
                          clean_up_raw_data: bool = True, verbose: bool = False):
    import tarfile
    destination = os.path.join(cache_dir, meta_info['name'], meta_info['name'] + meta_info['file_extension'])

    with tarfile.open(tmp_file_path) as file:
        file.extract('wikiann-da.bio', '/tmp/')

    _convert_wikiann_to_iob('/tmp/wikiann-da.bio', destination)

    os.remove(tmp_file_path)
    os.remove('/tmp/wikiann-da.bio')
