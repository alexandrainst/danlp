import time

from flair.data import Sentence, Token
from .utils import print_speed_performance

from danlp.datasets import DDT
from danlp.models import load_spacy_model, load_flair_ner_model, \
    load_bert_ner_model
from danlp.metrics import f1_report


def is_misc(ent: str):
    if len(ent) < 4:
        return False
    return ent[-4:] == 'MISC'


def remove_miscs(se: list):
    return [
        [entity if not is_misc(entity) else 'O' for entity in entities]
        for entities in se
    ]


# Load the DaNE data
_, _, test = DDT().load_as_simple_ner(predefined_splits=True)
sentences_tokens, sentences_entities = test

# Replace MISC with O for fair comparisons
sentences_entities = remove_miscs(sentences_entities)

num_sentences = len(sentences_tokens)
num_tokens = sum([len(s) for s in sentences_tokens])


def benchmark_polyglot_mdl():
    """
    Running ployglot requires these packages:
    # Morfessor==2.0.6
    # PyICU==2.4.2
    # pycld2==0.41
    # polyglot
    """
    from polyglot.tag import NEChunker
    from polyglot.text import WordList

    start = time.time()

    predictions = []
    for tokens in sentences_tokens:
        word_list = WordList(tokens, language='da')
        ne_chunker = NEChunker(lang='da')
        word_ent_tuples = list(ne_chunker.annotate(word_list))

        predictions.append([entity for word, entity in word_ent_tuples])
    print('polyglot:')
    print_speed_performance(start, num_sentences, num_tokens)
    assert len(predictions) == len(sentences_entities)

    print(f1_report(sentences_entities, remove_miscs(predictions), bio=True))

def benchmark_spacy_mdl():
    nlp = load_spacy_model()
    ner = nlp.entity

    predictions = []
    start = time.time()
    for token in sentences_tokens:
        doc = nlp.tokenizer.tokens_from_list(token)
        ner(doc)
        ents = []
        for t in doc:
            if t.ent_iob_ == 'O':
                ents.append(t.ent_iob_)
            else:
                ents.append(t.ent_iob_ + "-" + t.ent_type_)

        predictions.append(ents)
    print('spaCy:')
    print_speed_performance(start, num_sentences, num_tokens)

    assert len(predictions) == num_sentences
    
    print(f1_report(sentences_entities, remove_miscs(predictions), bio=True))


def benchmark_flair_mdl():
    tagger = load_flair_ner_model()

    start = time.time()

    flair_sentences = []
    for i, sentence in enumerate(sentences_tokens):
        flair_sentence = Sentence()

        for token_txt in sentence:
            flair_sentence.add_token(Token(token_txt))
        flair_sentences.append(flair_sentence)

    tagger.predict(flair_sentences, verbose=True)
    predictions = [[tok.tags['ner'].value for tok in fs] for fs in flair_sentences]
    print('Flair:')
    print_speed_performance(start, num_sentences, num_tokens)

    assert len(predictions) == num_sentences

    print(f1_report(sentences_entities, remove_miscs(predictions), bio=True))


def benchmark_bert_mdl():
    bert = load_bert_ner_model()

    start = time.time()

    predictions = []
    for i, sentence in enumerate(sentences_tokens):
        _, pred_ents = bert.predict(sentence)
        predictions.append(pred_ents)
    print('bert:')
    print_speed_performance(start, num_sentences, num_tokens)
    
    assert len(predictions) == num_sentences

    print(f1_report(sentences_entities, remove_miscs(predictions), bio=True))


if __name__ == '__main__':
    benchmark_polyglot_mdl()
    benchmark_spacy_mdl()
    benchmark_flair_mdl()
    benchmark_bert_mdl()
