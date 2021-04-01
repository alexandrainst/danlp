
import time
import os

import spacy

from danlp.datasets import DDT
from danlp.models import load_spacy_model

from utils import print_speed_performance, dependency_report

# load the data
ddt = DDT()

ccorpus_conll = ddt.load_as_conllu(predefined_splits=True)
deps_true = []
# the test set
sentences_tokens = []
for sent in ccorpus_conll[2]:
    sentences_tokens.append([token.form for token in sent._tokens])
    deps_true.append([(token.deprel.lower(),int(token.head)) for token in sent._tokens])
num_sentences = len(sentences_tokens)
num_tokens = sum([len(s) for s in sentences_tokens])


def benchmark_spacy_mdl():

    def normalize_spacy_head(i, hd):
        return 0 if i == hd else hd+1

    nlp = load_spacy_model()
    parser = nlp.parser
    
    start = time.time()

    deps_pred = []
    for sent in sentences_tokens:
        doc = nlp.tokenizer.tokens_from_list(sent)
        doc = parser(doc)

        deprels = []
        depheads = []
        for i, tok in enumerate(doc):
            deprels.append(tok.dep_.lower())
            depheads.append(normalize_spacy_head(i, tok.head.i))
        deps_pred.append([(r,h) for r,h in zip(deprels, depheads)])

    print('**Spacy model**')
    print_speed_performance(start, num_sentences, num_tokens)
    
    assert len(deps_pred)==num_sentences
    assert sum([len(s) for s in deps_pred])==num_tokens
    
    print(dependency_report(deps_true, deps_pred))

def benchmark_dacy_mdl(dacy_model="da_dacy_large_tft-0.0.0"):
    """
    an adaption of benchmark spacy model which is compatible with spacy v. 3

    running this requires:
    spacy >= 3.0.0
    spacy-transformers
    """
    def normalize_spacy_head(i, hd):
        return 0 if i == hd else hd+1

    from spacy.tokens import Doc
    import dacy
    nlp = dacy.load(dacy_model)
    trf = nlp.get_pipe('transformer')
    parser = nlp.get_pipe('parser')

    
    start = time.time()

    deps_pred = []
    for sent in sentences_tokens:
        doc = Doc(nlp.vocab, words=sent)
        doc = trf(doc)
        doc = parser(doc)

        deprels = []
        depheads = []
        for i, tok in enumerate(doc):
            deprels.append(tok.dep_.lower())
            depheads.append(normalize_spacy_head(i, tok.head.i))
        deps_pred.append([(r,h) for r,h in zip(deprels, depheads)])

    print('**Spacy model**')
    print_speed_performance(start, num_sentences, num_tokens)
    
    assert len(deps_pred)==num_sentences
    assert sum([len(s) for s in deps_pred])==num_tokens
    
    print(dependency_report(deps_true, deps_pred))


if __name__ == '__main__':
    benchmark_spacy_mdl()