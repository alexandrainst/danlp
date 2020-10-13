
import time
import os

from danlp.datasets import DDT
from danlp.models import load_spacy_chunking_model, get_noun_chunks
from danlp.metrics import f1_report

from spacy.tokens.doc import Doc
from spacy.gold import read_json_object


chunker = load_spacy_chunking_model()

def load_test_with_spacy(ddt):
    from spacy.cli.converters import conllu2json

    conll_path = os.path.join(ddt.dataset_dir, '{}.{}{}'.format(ddt.dataset_name, 'test', ddt.file_extension))

    file_as_json = {}
    with open(conll_path, 'r') as file:
        file_as_string = file.read()
        file_as_string = file_as_string.replace("name=", "").replace("|SpaceAfter=No", "")
        file_as_json = conllu2json(file_as_string)
    return read_json_object(file_as_json)


# load the data :
#      * convert to spaCy Docs format
#      * convert dependencies to (BIO) noun chunks
ddt = DDT()
corpus = load_test_with_spacy(ddt)
nlp = chunker.model
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

    start = time.time()

    chks_pred = []
    for sent in sentences_tokens:
        bio_chunks = chunker.predict(sent)
        chks_pred.append(bio_chunks)

    print('**Spacy model**')
    print("Made predictions on {} sentences and {} tokens in {}s".format(
    num_sentences, num_tokens, time.time() - start))

    assert len(chks_pred)==num_sentences
    assert sum([len(s) for s in chks_pred])==num_tokens

    print(f1_report(chks_true, chks_pred, bio=True))



if __name__ == '__main__':
    benchmark_spacy_mdl()