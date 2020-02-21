import time

from danlp.datasets import DDT
from danlp.models import load_spacy_model

from seqeval.metrics import classification_report


def is_misc(ent: str):
    if len(ent) < 4:
        return False
    return ent[-4:] == 'MISC'


def remove_miscs(se: list):
    return [
        [entity if not is_misc(entity) else 'O' for entity in entities]
        for entities in se
    ]


# Load the DaNe data
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

    print("Made predictions on {} sentences and {} tokens in {}s".format(
        num_sentences, num_tokens, time.time() - start))
    assert len(predictions) == len(sentences_entities)

    print(classification_report(sentences_entities, remove_miscs(predictions),
                                digits=4))


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

    print("Made predictions on {} sentences and {} tokens in {}s".format(
        num_sentences, num_tokens, time.time() - start)
    )

    assert len(predictions) == num_sentences

    print(classification_report(sentences_entities, remove_miscs(predictions),
                                digits=4))


if __name__ == '__main__':
    benchmark_spacy_mdl()
    benchmark_polyglot_mdl()
