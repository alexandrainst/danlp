
import time

from flair.data import Sentence, Token

from danlp.datasets import DDT
from danlp.models import load_spacy_model, load_flair_pos_model

from seqeval.metrics import classification_report

# bechmarking polyglotmodel requires
from polyglot.tag import POSTagger
from polyglot.text import WordList



# load the data
ddt = DDT()

corpus_flair =  ddt.load_with_flair()
tags_true = [[tok.tags['pos'].value for tok in fs] for fs in corpus_flair.test]
num_sentences = len(tags_true)
num_tokens = sum([len(s) for s in tags_true])

ccorpus_conll =ddt.load_as_conllu(predefined_splits=True)
# the test set
sentences_tokens = []
for sent in ccorpus_conll[2]:
    sentences_tokens.append([token.form for token in sent._tokens])
    
    
def benchmark_flair_mdl():
    tagger = load_flair_pos_model()
    
    start = time.time()
    tagger.predict(corpus_flair.test)
    tags_pred = [[tok.tags['upos'].value for tok in fs] for fs in corpus_flair.test]
    
    print('**Flair model** ')
    print("Made predictions on {} sentences and {} tokens in {}s".format(
    num_sentences, num_tokens, time.time() - start))
    
    assert len(tags_pred)==num_sentences
    assert sum([len(s) for s in tags_pred])==num_tokens
    
    
    print(classification_report(tags_true, tags_pred,
                                    digits=4))
    
    
    
    
def benchmark_spacy_mdl():
    nlp = load_spacy_model()
    tagger = nlp.tagger
    
    start = time.time()

    tags_pred = []
    for sent in sentences_tokens:
        doc = nlp.tokenizer.tokens_from_list(sent)
        doc=tagger (doc)

        tags = []
        for tok in doc:

            tags.append(tok.pos_)

        tags_pred.append(tags)
    print('**Spacy model**')
    print("Made predictions on {} sentences and {} tokens in {}s".format(
    num_sentences, num_tokens, time.time() - start))
    
    assert len(tags_pred)==num_sentences
    assert sum([len(s) for s in tags_pred])==num_tokens
    
    
    print(classification_report(tags_true, tags_pred,
                                    digits=4))
    

    
def benchmark_polyglot_mdl():
    """
    Running ployglot requires these packages:
    # Morfessor==2.0.6
    # PyICU==2.4.2
    # pycld2==0.41
    # polyglot
    
    """
    
    start = time.time()

    tags_pred = []
    for tokens in  sentences_tokens:
        word_list = WordList(tokens, language='da')
        ne_chunker =  POSTagger(lang='da')
        word_ent_tuples = list(ne_chunker.annotate(word_list))

        tags_pred.append([entity for word, entity in word_ent_tuples])
    print('**Polyglot model**')
    print("Made predictions on {} sentences and {} tokens in {}s".format(
    num_sentences, num_tokens, time.time() - start))
    
    assert len(tags_pred)==num_sentences
    assert sum([len(s) for s in tags_pred])==num_tokens
    
    
    print(classification_report(tags_true, tags_pred,
                                    digits=4))
    
    
    
    
    
    
if __name__ == '__main__':
    benchmark_polyglot_mdl()
    benchmark_spacy_mdl()
    benchmark_flair_mdl()