
import time
import os

from danlp.datasets import DDT
from danlp.models import load_spacy_model

from seqeval.metrics import classification_report

from spacy.tokens.doc import Doc
from spacy.gold import read_json_object

import numpy as np

nlp = load_spacy_model()


def load_test_with_spacy(ddt):
    from spacy.cli.converters import conllu2json

    conll_path = os.path.join(ddt.dataset_dir, '{}.{}{}'.format(ddt.dataset_name, 'test', ddt.file_extension))

    file_as_json = {}
    with open(conll_path, 'r') as file:
        file_as_string = file.read()
        file_as_string = file_as_string.replace("name=", "").replace("|SpaceAfter=No", "")
        file_as_json = conllu2json(file_as_string)
    return read_json_object(file_as_json)


left_labels = ["det", "fixed", "nmod:poss", "amod", "flat", "goeswith", "nummod", "appos"]
right_labels = ["fixed", "nmod:poss", "amod", "flat", "goeswith", "nummod", "appos"]
stop_labels = ["punct"]

np_label = "NP"

def get_noun_chunks(doc, bio=False, nested=False):

    def is_verb_token(tok):
        return tok.pos_ in ['VERB', 'AUX']

    def next_token(tok):
        try:
            return tok.nbor()
        except IndexError:
            return None

    def get_left_bound(doc, root):
        left_bound = root
        for tok in reversed(list(root.lefts)):
            if tok.dep_ in left_labels:
                left_bound = tok
        return left_bound

    def get_right_bound(doc, root):
        right_bound = root
        for tok in root.rights:
            if tok.dep_ in right_labels:
                right = get_right_bound(doc, tok)
                if list(
                    filter(
                        lambda t: is_verb_token(t) or t.dep_ in stop_labels,
                        doc[root.i : right.i],
                    )
                ):
                    break
                else:
                    right_bound = right
        return right_bound

    def get_bounds(doc, root):
        return get_left_bound(doc, root), get_right_bound(doc, root)


    chunks = []
    for token in doc:
        if token.pos_ in ["PROPN", "NOUN", "PRON"]:
            left, right = get_bounds(doc, token)
            chunks.append((left.i, right.i + 1, np_label))
            token = right

    is_chunk = [True for _ in chunks]
    if not nested:
        # remove nested chunks
        for i, i_chunk in enumerate(chunks[:-1]):
            i_left, i_right, _ = i_chunk 
            for j, j_chunk in enumerate(chunks[i+1:], start=i+1):
                j_left, j_right, _ = j_chunk
                if j_left <= i_left < i_right <= j_right:
                    is_chunk[i] = False
                if i_left <= j_left < j_right <= i_right:
                    is_chunk[j] = False

    final_chunks = [c for c, ischk in zip(chunks, is_chunk) if ischk]
    return chunks2bio(final_chunks, len(doc)) if bio else final_chunks

def chunks2bio(chunks, sent_len):
    bio_tags = ['O'] * sent_len
    for (start, end, label) in chunks:
        bio_tags[start] = 'B-'+label
        for j in range(start+1, end):
            bio_tags[j] = 'I-'+label
    return bio_tags


# load the data :
#      * convert to spaCy Docs format
#      * convert dependencies to (BIO) noun chunks
ddt = DDT()
corpus = load_test_with_spacy(ddt)
sentences_tokens = []
chks_true = []
for jobj in corpus:
    for sentence in jobj[1]:
        sentence = sentence[0]
        tokens = sentence[1]
        sentences_tokens.append(tokens)
        doc = Doc(nlp.vocab, words=tokens)
        for i, t in enumerate(doc):
            t.head =  doc[sentence[3][i]]
            t.pos = nlp.vocab.strings.add(sentence[2][i])
            t.dep = nlp.vocab.strings.add(sentence[4][i])
        bio_chks = get_noun_chunks(doc, bio=True)
        chks_true.append(bio_chks)

num_sentences = len(sentences_tokens)
num_tokens = sum([len(s) for s in sentences_tokens])



def benchmark_spacy_mdl():

    parser = nlp.parser
    tagger = nlp.tagger

    start = time.time()

    chks_pred = []
    for sent in sentences_tokens:
        doc = nlp.tokenizer.tokens_from_list(sent)
        doc = tagger(doc)
        doc = parser(doc)

        bio_chunks = get_noun_chunks(doc, bio=True)
        chks_pred.append(bio_chunks)

    print('**Spacy model**')
    print("Made predictions on {} sentences and {} tokens in {}s".format(
    num_sentences, num_tokens, time.time() - start))

    assert len(chks_pred)==num_sentences
    assert sum([len(s) for s in chks_pred])==num_tokens

    print(classification_report(chks_true, chks_pred, digits=4))



if __name__ == '__main__':
    benchmark_spacy_mdl()