import os
import random
import shutil
import string
from typing import Union


def random_string(length: int = 12):
    letters = string.ascii_lowercase

    return ''.join(random.choice(letters) for i in range(length))


def read_simple_ner_dataset(file_paths: Union[list, str], token_idx: int = 0,
                            entity_idx: int = 1):
    """
    Reads a dataset in the simple NER format similar to
    the CoNLL 2003 NER format.

    :param file_paths: one or more filepaths
    :param entity_idx:
    :param token_idx:
    :return: list of sentences, ents
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    sentences = []
    entities = []

    for path in file_paths:
        with open(path, 'r') as f:
            sentence = []
            ent = []
            for line in f:
                if line == "\n":
                    assert len(sentence) == len(ent)
                    sentences.append(sentence)
                    entities.append(ent)

                    sentence = []
                    ent = []
                elif not line.startswith("#"):
                    sentence.append(line.split()[token_idx])
                    ent.append(line.split()[entity_idx].strip())

    return sentences, entities


def write_simple_ner_dataset(sentences: list, entitites: list, file_path: str):
    """
    Writes a dataset in the simple NER format similar to
    the CoNLL 2003 NER format.

    :param sentences:
    :param entitites:
    :param file_path:
    """
    with open(file_path, "w", encoding="utf8") as f:
        for ss, es in zip(sentences, entitites):
            for s, e in zip(ss, es):
                f.write("{} {}\n".format(s, e))

            f.write("\n")


def extract_single_file_from_zip(cache_dir: str, file_in_zip: str, dest_full_path, zip_file):
    # To not have name conflicts

    tmp_path = os.path.join(cache_dir, ''.join(random_string()))

    outpath = zip_file.extract(file_in_zip, path=tmp_path)
    os.rename(outpath, dest_full_path)

    shutil.rmtree(tmp_path)
