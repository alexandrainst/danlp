
import time
from utils import print_speed_performance, accuracy_report

from flair.data import Sentence, Token

from danlp.datasets import DDT
from danlp.models import load_spacy_model, load_flair_pos_model

# benchmarking polyglotmodel requires
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
    print_speed_performance(start, num_sentences, num_tokens)
    
    assert len(tags_pred)==num_sentences
    assert sum([len(s) for s in tags_pred])==num_tokens
    
    print(accuracy_report(tags_true, tags_pred))
    
    
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
    print_speed_performance(start, num_sentences, num_tokens)
    
    assert len(tags_pred)==num_sentences
    assert sum([len(s) for s in tags_pred])==num_tokens
    
    print(accuracy_report(tags_true, tags_pred))


auxiliary_verbs = ["være", "er", "var", "været"]
auxiliary_verbs += ["have", "har", "havde", "haft"]
auxiliary_verbs += ["kunne", "kan", "kunnet"]
auxiliary_verbs += ["ville", "vil", "villet"]
auxiliary_verbs += ["skulle", "skal", "skullet"]
auxiliary_verbs += ["måtte", "må", "måttet"]
auxiliary_verbs += ["burde", "bør", "burdet"]

def benchmark_polyglot_mdl(corrected_output=False):
    """
    Running polyglot requires these packages:
    # Morfessor==2.0.6
    # PyICU==2.4.2
    # pycld2==0.41
    # polyglot
    
    """

    def udify_tag(tag, word):
        if tag == "CONJ":
            return "CCONJ"
        if tag == "VERB" and word in auxiliary_verbs:
            return "AUX"
        return tag
    
    start = time.time()

    tags_pred = []
    for tokens in  sentences_tokens:
        word_list = WordList(tokens, language='da')
        tagger =  POSTagger(lang='da')
        word_tag_tuples = list(tagger.annotate(word_list))
        tags_pred.append([udify_tag(tag, word) if corrected_output else tag for word, tag in word_tag_tuples])
    print('**Polyglot model'+(' (corrected output) ' if corrected_output else '')+'**')
    print_speed_performance(start, num_sentences, num_tokens)

    assert len(tags_pred)==num_sentences
    assert sum([len(s) for s in tags_pred])==num_tokens

    print(accuracy_report(tags_true, tags_pred))




if __name__ == '__main__':
    benchmark_polyglot_mdl()
    benchmark_polyglot_mdl(corrected_output=True)
    benchmark_spacy_mdl()
    benchmark_flair_mdl()