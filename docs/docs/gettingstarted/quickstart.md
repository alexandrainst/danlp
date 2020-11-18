Quick start
===========

Once you have [installed](installation.md) the DaNLP package, you can use it in your python project using `import danlp`. 

You will find the main functions through the `models` and `datasets` modules -- see the library documentation for more details about how to use the different functions for loading models and datasets. 
For analysing texts in Danish, you will primarily need to import functions from `danlp.models` in order to load and use our pre-trained models. 

The DaNLP package provides you with several models for different NLP tasks using different frameworks. 
On this section, you will have a quick tour of the main functions of the DaNLP package. 
For a more detailed description of the tasks and frameworks, follow the links to the documentation: 

*  [Embedding of text](../tasks/embeddings.md) with flair, spaCy or Gensim
*  [Part of speech tagging](../tasks/pos.md) (POS) with spaCy or flair
*  [Named Entity Recognition](../tasks/ner.md) (NER) with spaCy, flair or BERT
*  [Sentiment Analysis](../tasks/sentiment_analysis.md) with spaCy or BERT
*  [Dependency parsing and NP-chunking](../tasks/dependency.md) with spaCy


## All-in-one with the spaCy models

To quickly get started with DaNLP and try out different NLP tasks, you can use the spaCy model ([see also](../frameworks/spacy.md)). The main advantages of the spaCy model is that it is fast and it includes most of the basic NLP tasks that you need for pre-processing texts in Danish. 

The main functions are:  

* `load_spacy_model` for loading a spaCy model for POS, NER and dependency parsing or a spaCy sentiment model
* `load_spacy_chunking_model` for loading a wrapper around the spaCy model with which you can deduce NP-chunks from dependency parses

### Pre-processing tasks

Perform [Part-of-Speech tagging](../tasks/pos.md), [Named Entity Recognition](../tasks/ner.md) and [dependency parsing](../tasks/dependency.md) at the same time with the DaNLP spaCy model.
Here is a snippet to quickly getting started: 

```python
# Import the load function
from danlp.models import load_spacy_model

# Download and load the spaCy model using the DaNLP wrapper function
nlp = load_spacy_model()

# Parse the text using the spaCy model 
# it creates a spaCy Doc object
doc = nlp("Jeg er en sætning, der skal analyseres")

# prepare some pretty printing
features = ['Text','POS', 'Dep']
head_format ="\033[1m{!s:>11}\033[0m" * (len(features) )
row_format ="{!s:>11}" * (len(features) )

print(head_format.format(*features))
# printing for each token in the docs the pos and dep features
for token in doc:
    print(row_format.format(token.text, token.pos_, token.dep_))
    
```

For NP-chunking you can use the `load_spacy_chunking_model`. 
The spaCy chunking model includes the spaCy model -- which can be used as previously described. 

```python
from danlp.models import load_spacy_chunking_model

# text to process
text = 'Et syntagme er en gruppe af ord, der hænger sammen'

# Load the chunker using the DaNLP wrapper
chunker = load_spacy_chunking_model()
# Applying the spaCy model for parsing the sentence
# and deducing NP-chunks
np_chunks = chunker.predict(text, bio=False)

nlp = chunker.model
doc = nlp(text)

# print the chunks
for (start_id, end_id, _) in np_chunks: 
    print(doc[start_id:end_id])
```


### Sentiment analysis

With the spaCy sentiment model, you can predict whether a sentence is perceived positive, negative or neutral. 
For loading and using the spaCy sentiment analyser, follow these steps: 

```python
from danlp.models import load_spacy_model

# Download and load the spaCy sentiment model using the DaNLP wrapper function
nlpS = load_spacy_model(textcat='sentiment', vectorError=True)

text = "Jeg er meget glad med DaNLP"

# analyse the text using the spaCy sentiment analyser
doc = nlpS(text)

# print the most probable category among 'positiv', 'negativ' or 'neutral'
print(max(doc.cats))
```


## Sequence labelling with flair

For part-of-speech tagging and named entity recognition, you also have the possibility to use flair. 
If you value precision rather than speed, we would recommend you to use the flair models (or BERT NER, next section). 

Perform POS tagging or NER using the DaNLP flair models that you can load through the following functions:

* `load_flair_pos_model`
* `load_flair_ner_model`

Use the following snippet to try out the flair POS model. Note that the text should be pre-tokenized. 

```python
from danlp.models import load_flair_pos_model
from flair.data import Sentence

text = "Hans har en lille sort kat ."
sentence = Sentence(text)

tagger = load_flair_pos_model()

tagger.predict(sentence)

for tok in sentence.tokens:
    print(tok.text, tok.get_tag('upos').value)

```

You can use the flair NER model in a similar way. 


```python
from danlp.models import load_flair_ner_model
from flair.data import Sentence

text = "Hans bor i København"
sentence = Sentence(text)

tagger = load_flair_ner_model()

tagger.predict(sentence)

for tok in sentence.tokens:
    print(tok.text, tok.get_tag('ner').value)
```

## Deep NLP with BERT {#bert}

### NER with BERT

You can also perform NER with BERT. Load the DaNLP model with `load_bert_ner_model` and try out the following snippet: 

```python
from danlp.models import load_bert_ner_model
bert = load_bert_ner_model()
# Get lists of tokens and labesl in IBO format
tokens, labels = bert.predict("Jens Peter Hansen kommer fra Danmark")
print(" ".join(["{}/{}".format(tok,lbl) for tok,lbl in zip(tokens,labels)]))

# To get a "right" tokenization provide it your self (SpaCy can be used for this) by providing a a list of tokens
# With this option, output can also be choosen to be a dict with tags and position instead of IBO format
tekst_tokenized = ['Han', 'hedder', 'Anders', 'And', 'Andersen', 'og', 'bor', 'i', 'Århus', 'C']
bert.predict(tekst_tokenized, IOBformat=False)
"""
{'text': 'Han hedder Anders And Andersen og bor i Århus C',
 'entities': [{'type': 'PER','text':'Anders And Andersen','start_pos': 11,'end_pos': 30},
  {'type': 'LOC', 'text': 'Århus C', 'start_pos': 40, 'end_pos': 47}]}
"""

```


### Classification with BERT

BERT is well suited for classification tasks. You can load the DaNLP sentiment classification BERT models with:

* `load_bert_emotion_model`
* `load_bert_tone_model`


With the BERT Emotion model you can classify sentences among eight emotions: 

* `Glæde/Sindsro`
* `Tillid/Accept`
* `Forventning/Interrese`
* `Overasket/Målløs`
* `Vrede/Irritation`
* `Foragt/Modvilje`
* `Sorg/trist`
* `Frygt/Bekymret`

Following is an example of how to use the BERT Emotion model:

```python
from danlp.models import load_bert_emotion_model
classifier = load_bert_emotion_model()

# using the classifier
classifier.predict('jeg ejer en bil')
''''No emotion''''
classifier.predict('jeg ejer en rød bil og det er en god bil')
''''Tillid/Accept''''
classifier.predict('jeg ejer en rød bil men den er gået i stykker')
''''Sorg/trist''''

# Get probabilities and matching classes
probas = classifier.predict_proba('jeg ejer en rød bil men den er gået i stykker', no_emotion=False)[0]
classes = classifier._classes()[0]

for c, p in zip(classes, probas):
    print(c, ':', p)
```

With the BERT Tone model, you can predict the tone (`objective` or `subjective`) or the polarity (`positive`, `negative` or `neutral`) of sentences. 


```python
from danlp.models import load_bert_tone_model
classifier = load_bert_tone_model()

text = 'Analysen viser, at økonomien bliver forfærdelig dårlig'

# using the classifier
prediction = classifier.predict(text)
print("Tone: ", prediction['analytic'])
print("Polarity: ", prediction['polarity'])

# Get probabilities and matching classes
probas = classifier.predict_proba(text)[0]
classes = classifier._classes()[0]

for c, p in zip(classes, probas):
    print(c, ':', p)
```