import os
import random
import shutil
import string
from typing import Union
import requests

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
    if not os.path.exists(dest_full_path):
        os.rename(outpath, dest_full_path)

    shutil.rmtree(tmp_path)


def get_wikidata_qids_from_entity(entity_text):
    """
    Use Wikidata search API to get a list of potential Wikidata QIDs for an entity.

    :param str entity_text: raw text
    :return: list of QIDs (strings)
    """

    query = entity_text.strip().replace(" ", "+")
    url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&search="+query+"&language=da&format=json&limit=50"
    res = requests.get(url).json()

    return [search['id'] for search in res['search']]
    

def get_label_from_wikidata_qid(qid):    
    """
    Use Wikidata API to get the label of an entity refered by its Wikidata QID.

    :param str qid: Wikidata QID (or PID)
    :return str: label 
    """

    url = "https://www.wikidata.org/w/api.php?action=wbgetentities&ids="+qid+"&props=labels%7Cclaims&languages=da&languagefallback=en&formatversion=2&format=json"
    try:
        r = requests.get(url, timeout=10)
        out = r.json()
    except:
        out = {}
    
    try: 
        return out['entities'][qid]['labels']['da']['value']
    except KeyError:
        return qid


def get_kg_context_from_wikidata_qid(qid: str):
    """
    Use Wikidata API to get the description and list of properties
    of an entity refered by its Wikidata QID.

    :param str qid: Wikidata QID (or PID)
    :return str: label 
    """

    url = "https://www.wikidata.org/w/api.php?action=wbgetentities&ids="+qid+"&props=descriptions%7Cclaims&languages=da&languagefallback=en&formatversion=2&format=json"
    response = requests.get(url).json()

    try:
        description = response['entities'][qid]['descriptions']['da']['value']
    except: 
        description = None

    claims = response['entities'][qid]['claims']

    knowledge_graph = []
    for claim in claims.keys():
        prop = get_label_from_wikidata_qid(claim)
        for d in claims[claim]:
            try:
                claim_type = d['mainsnak']['datavalue']['type']
                if claim_type == 'string':
                    label = d['mainsnak']['datavalue']['value']
                elif claim_type == 'time':
                    label = d['mainsnak']['datavalue']['value']['time']
                elif claim_type == 'monolingualtext':
                    label = d['mainsnak']['datavalue']['value']['text']
                elif claim_type == 'quantity':
                    label = d['mainsnak']['datavalue']['value']['amount']
                elif claim_type == 'wikibase-entityid': #if entityid look up
                    val = d['mainsnak']['datavalue']['value']['id']
                    label = get_label_from_wikidata_qid(val)
                else:
                    continue
                knowledge_graph.append([prop, label])
            except KeyError:
                continue

    return knowledge_graph, description

