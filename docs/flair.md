Flair
=====

The [flair](https://github.com/flairNLP/flair) framework from Zalando is based on the paper [Akbik et. al (2018)](<https://alanakbik.github.io/papers/coling2018.pdf>). 


Through the DaNLP package, we provide a pre-trained Part-of-Speech tagger and Named Entity recognizer using the flair framework. 
The models have been trained on the [Danish Dependency Treebank](datasets.md#dane) and use fastText word embeddings and [flair contextual word embeddings](models/embeddings.md#flair-embeddings) trained on data from Wikipedia and EuroParl corpus.
The code for training can be found on flair's GitHub, and the following parameters are set:
`learning_rate=1`, `mini_batch_size=32`, `max_epochs=150`, `hidden_size=256`.


## One sentence at a time

For Part-of-Speech tagging and Named Entity Recognition, it is possible to analyse one sentence at a time using the `Sentence` class of the flair framework. 

Please note that the text should be tokenized before hand. 

Here is a snippet for using the part-of-speech tagger which can be loaded using the DaNLP `load_flair_pos_model` function.  

```python
from danlp.models import load_flair_pos_model
from flair.data import Sentence

text = "Morten bor i København tæt på Kongens Nytorv"
sentence = Sentence(text)

tagger = load_flair_pos_model()

tagger.predict(sentence)

for tok in sentence.tokens:
    print(tok.text, tok.get_tag('upos').value)

```

In a similar way, you can load and use the DaNLP Named Entity Recognition model using the `load_flair_ner_model` function.

```python
from danlp.models import load_flair_ner_model
from flair.data import Sentence

text = "Morten bor i København tæt på Kongens Nytorv"
sentence = Sentence(text)

tagger = load_flair_ner_model()

tagger.predict(sentence)

for tok in sentence.tokens:
    print(tok.text, tok.get_tag('ner').value)

```


## Dataset analysis


If you want to analyze an entire dataset you can either use one of the DaNLP functions to load the DDT or the WikiAnn, or create a list of flair `Sentence`. 

### DaNLP datasets

You can load the DDT as follow:

```python
from danlp.datasets import DDT
ddt = DDT()
# load the DDT
flair_corpus = ddt.load_with_flair()

# you can access the train, test or dev part of the dataset
flair_train = flair_corpus.train
flair_test = flair_corpus.test
flair_dev = flair_corpus.dev

# to get the list of UPOS tags for each sentence
pos_tags = [[tok.tags['upos'].value for tok in fs] for fs in flair_test]
# to get the list of NER tags for each sentence (BIO format)
ner_tags = [[tok.tags['ner'].value for tok in fs] for fs in flair_test]
# to get the list of tokens for each sentence
tokens = [[tok.text for tok in fs] for fs in flair_test]

# you can use the loaded datasets 
# to parse with the danlp POS or NER models
tagger.predict(flair_test)

```

Or the WikiAnn: 

```python
from danlp.datasets import WikiAnn
wikiann = WikiAnn()
# load the WikiAnn dataset
flair_corpus = wikiann.load_with_flair()
```

### Your dataset

From your own list of sentences (pre-tokenized) you can build a list of flair `Sentence` -- to use as previously described with the DaNLP datasets. 

Here is an example with the POS model:

```python
from danlp.models import load_flair_pos_model
from flair.data import Sentence, Token

# loading the POS flair model
tagger = load_flair_pos_model()

# your sentences (list of lists of tokens)
my_sentences = [[...], [...], ...]

flair_sentences = []
for sent in my_sentences:
    flair_sent = Sentence()
    for tok in sent:
        flair_sent.add_token(Token(tok))
    flair_sentences.append(flair_sent)

tagger.predict(flair_sentences)

for sentence in flair_sentences: 
    print(" ".join(["{}/{}".format(t.text, t.get_tag('upos').value) for t in sentence.tokens]))
```