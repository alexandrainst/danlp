Part of Speech Tagging
======================
This section is concerned with public available Part of Speech (POS) taggers in Danish. 

| Model | Train Data | License | Trained by | Tags | DaNLP |
|-------|-------|-------|-------|-------|-------|
| [Polyglot](https://github.com/alexandrainst/danlp/blob/master/docs/models/pos.md#polyglot) | [Danish Dependency Treebank](<https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane>) [Al-Rfou et al. (2013)] | GPLv3 license | Polyglot | 17  Universal part of speech | ❌ |
| [Flair](https://github.com/alexandrainst/danlp/blob/master/docs/models/pos.md#flair) | [Danish Dependency Treebank](<https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane>) | MIT | Alexandra Instittut | 17  Universal part of speech | ✔️ |
| [SpaCy](https://github.com/alexandrainst/danlp/blob/master/docs/models/pos.md#spacy) | [Danish Dependency Treebank](<https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane>) | MIT | Alexandra Instittut | 17  Universal part of speech | ✔️ |

The Danish UD treebank  uses 17 [universal part of speech tags](<https://universaldependencies.org/u/pos/index.html>):

`ADJ`: Adjective, `ADP`: Adposition , `ADV`: Adverb, `AUX`: Auxiliary verb, `CCONJ`: Coordinating conjunction, `DET`: Determiner, `INTJ`: Interjection, `NOUN`: Noun, `NUM`: Numeral, `PART`: Particle `PRON`: Pronoun `PROPN`: Proper noun `PUNCT`: Punctuation `SCONJ`: Subordinating conjunction `SYM`: Symbol `VERB`: Verb `X`: Other

A medium blog using Part of Speech tagging on Danish, can be found  [here](<https://medium.com/danlp/i-klasse-med-kierkegaard-eller-historien-om-det-fede-ved-at-en-computer-kan-finde-ordklasser-189774695f3b>).

![](../imgs/postag_eksempel.gif)

##### 🔧 Flair

This project provides a trained part of speech tagging model for Danish using the [Flair](<https://github.com/flairNLP/flair>) framework from Zalando, based on the paper [Akbik et. al (2018)](<https://alanakbik.github.io/papers/coling2018.pdf>). The model is trained using the data [Danish Dependency Treebank](<https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane>)  and by using FastText word embeddings and Flair contextual word embeddings trained in this project on data from Wikipedia and EuroParl corpus, see [here](<https://github.com/alexandrainst/danlp/blob/master/docs/models/embeddings.md>).

The code for training can be found on Flairs GitHub, and the following parameters are set:
`learning_rate=1`, `mini_batch_size=32`, `max_epochs=150`, `hidden_size=256`.

The flair pos tagger can be used by loading  it with the  `load_flair_pos_model` method. Please note that the text should be tokenized before hand, this can for example be done using spacy. 

```python
from danlp.models import load_flair_pos_model
from flair.data import Sentence

# Load the POS tagger using the DaNLP wrapper
tagger = load_flair_pos_model()

# Using the flair POS tagger
sentence = Sentence('Jeg hopper på en bil , som er rød sammen med Niels .') 
tagger.predict(sentence) 
print(sentence.to_tagged_string())

# Example
'''Jeg <PRON> hopper <VERB> på <ADP> en <DET> bil <NOUN> , <PUNCT> som <ADP> er <AUX> rød <ADJ> sammen <ADV> med <ADP> Niels <PROPN> . <PUNCT>
'''

```



##### 🔧 SpaCy

Read more about the spaCy model in the dedicated [spaCy docs](<https://github.com/alexandrainst/danlp/blob/master/docs/spacy.md>) , it has also been trained using the [Danish Dependency Treebank](<https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane>) data. 

Below is a small getting started snippet for using the Spacy pos tagger:

```python
from danlp.models import load_spacy_model

#Load the POS tagger using the DaNLP wrapper
nlp = load_spacy_model()

# Using the spaCy POS tagger
doc = nlp('Jeg hopper på en bil, som er rød sammen med Niels.')
pred=''
for token in doc:
    pred += '{} <{}> '.format(token.text, token.pos_)
print(pred)

# Example
''' Jeg <PRON> hopper <VERB> på <ADP> en <DET> bil <NOUN> , <PUNCT> som <ADP> er <AUX> rød <ADJ> sammen <ADV> med <ADP> Niels <PROPN> . <PUNCT> 
 '''
```

##### Polyglot

Read more about the polyglot model [here](<https://polyglot.readthedocs.io/en/latest/POS.html>), and in the original paper [Al-Rfou et al. (2013)](https://www.aclweb.org/anthology/W13-3520). 

## 📈 Benchmarks

Accuracy scores is reported below and can be reproduced using `pos_benchmarks.py` in the [example](<https://github.com/alexandrainst/danlp/tree/master/examples>) folder, where the details score from each class is calculated.

#### DaNLP

| Model                       | Accuracy   |
|-----------------------------|------------|
| Flair                       | **97.97**  |
| SpaCy                       | 96.15      |

#### Polyglot

The tags predicted with the polyglot model differ slightly from the universal PoS-tags. The model predicts :
* `CONJ` instead of `CCONJ`
* `VERB` instead of `AUX` for the auxiliary and modal verbs (i.e. `være`, `have`, `kunne`, `ville`, `skulle`, `måtte`, `burde`)

We calculated the scores for the original predictions and for the corrected version.

| Model                       | Accuracy   |
| --------                    | ---------- |
| Polyglot                    | 76.76      |
| Polyglot (corrected output) | 83.4       |




## 🎓 References 
- Rami Al-Rfou, Bryan Perozzi, and Steven Skiena. 2013. [Polyglot: Distributed Word Representations for Multilingual NLP](https://www.aclweb.org/anthology/W13-3520). In **CoNLL**.
- Alan Akbik, Duncan Blythe, and Roland Vollgraf. 2018. [Contextual String Embeddings for Sequence Labeling](https://alanakbik.github.io/papers/coling2018.pdf). In **COLING**.

  
