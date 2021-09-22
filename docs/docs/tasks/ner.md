Named Entity Recognition
========================
Named Entity Recognition (NER) is the task of extracting named entities in a raw text. 
Common entity types are locations, organizations and persons. Currently a few
tools are available for NER in Danish. Popular models for NER
([BERT](https://huggingface.co/transformers/index.html),
[Flair](https://github.com/flairNLP/flair) and [spaCy](https://spacy.io/))
are continuously trained on the newest available named entity datasets such as DaNE
and made available through the DaNLP library.

| Model                                                                             | Train Data                                                            | Maintainer                     | Tags          | DaNLP |
|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------|--------------------------------|---------------|-------|
| [BERT](#bert)                                                                     | [DaNE](../datasets.md#dane)                                           | Alexandra Institute            | PER, ORG, LOC | ✔     |
| [Flair](#flair)                                                                   | [DaNE](../datasets.md#dane)                                           | Alexandra Institute            | PER, ORG, LOC | ✔     |
| [spaCy](#spacy)                                                                   | [DaNE](../datasets.md#dane)                                           | Alexandra Institute            | PER, ORG, LOC | ✔     |
| [Polyglot](https://polyglot.readthedocs.io/en/latest/NamedEntityRecognition.html) | Wikipedia                                                             | Polyglot                       | PER, ORG, LOC | ❌     |
| [daner](https://github.com/ITUnlp/daner)                                          | [Derczynski et al. (2014)](https://www.aclweb.org/anthology/E14-2016) | [ITU NLP](https://nlp.itu.dk/) | PER, ORG, LOC | ❌     |
| [NERDA](https://github.com/ebanalyse/NERDA/)                                      | [DaNE](../datasets.md#dane)                                           | Ekstra Bladet                  | PER, ORG, LOC | ❌     |
| [DaCy](#dacy)                                      | [DaNE](../datasets.md#dane)                                           | [Center for Humanities Computing Aarhus](http://chcaa.io/#/), [K. Enevoldsen ](http://kennethenevoldsen.com)                 | PER, ORG, LOC | (✔)    |

### Use cases

NER is one of the most famous NLP tasks used in the industry, probably because its use cases are pretty straightforward. 
It can be used in many systems, by itself or in combination with other NLP models. 
For instance, the extraction of entities from text can be used for : 
 * classifying / indexing documents (e.g articles for news providers) and then
 * recommending similar content (e.g. news articles)
 * customer support (e.g. for tagging tickets)
 * analysing feedback from customers (product reviews)
 * speeding up search engines 
 * extracting information (e.g from emails)
 * building a structured database or a knowledge graph (see our example [tutorial](https://github.com/alexandrainst/danlp/blob/master/examples/tutorials/example_knowledge_graph.ipynb)) from a corpus
 * anonymizing documents. 


## Models

### 🔧 BERT {#bert}
The BERT [(Devlin et al. 2019)](https://www.aclweb.org/anthology/N19-1423/) NER model is based on the pre-trained [Danish BERT](https://github.com/botxo/nordic_bert) representations by BotXO which 
has been finetuned on the [DaNE](../datasets.md#dane) 
dataset [(Hvingelby et al. 2020)](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.565.pdf). The finetuning has been done using the [Transformers](https://github.com/huggingface/transformers) library from HuggingFace.

The BERT NER model can be loaded with the `load_bert_ner_model()` method. Please notice that it can maximum take 512 tokens as input at a time. For longer text sequences split before hand, for example using sentence boundary detection (e.g. by using the [spacy model](../frameworks/spacy.md ).) 

```python
from danlp.models import load_bert_ner_model
bert = load_bert_ner_model()
# Get lists of tokens and labels in BIO format
tokens, labels = bert.predict("Jens Peter Hansen kommer fra Danmark")
print(" ".join(["{}/{}".format(tok,lbl) for tok,lbl in zip(tokens,labels)]))

# To get a correct tokenization, you have to provide it yourself to BERT  by providing a list of tokens
# (for example SpaCy can be used for tokenization)
# With this option, output can also be choosen to be a dict with tags and position instead of BIO format
tekst_tokenized = ['Han', 'hedder', 'Anders', 'And', 'Andersen', 'og', 'bor', 'i', 'Århus', 'C']
bert.predict(tekst_tokenized, IOBformat=False)
"""
{'text': 'Han hedder Anders And Andersen og bor i Århus C',
 'entities': [{'type': 'PER','text':'Anders And Andersen','start_pos': 11,'end_pos': 30},
  {'type': 'LOC', 'text': 'Århus C', 'start_pos': 40, 'end_pos': 47}]}
"""
```


### 🔧 Flair {#flair}
The Flair [(Akbik et al. 2018)](https://www.aclweb.org/anthology/C18-1139/) NER model
uses pretrained [Flair embeddings](embeddings.md#flair-embeddings)
in combination with fastText word embeddings. The model is trained using the [Flair](https://github.com/flairNLP/flair)
 library on the the [DaNE](../datasets.md#dane) dataset.

The Flair NER model can be used with DaNLP using the `load_flair_ner_model()` method.
```python
from danlp.models import load_flair_ner_model
from flair.data import Sentence

# Load the NER tagger using the DaNLP wrapper
flair_model = load_flair_ner_model()

# Using the flair NER tagger
sentence = Sentence('Jens Peter Hansen kommer fra Danmark') 
flair_model.predict(sentence) 
print(sentence.to_tagged_string())
```

### 🔧 spaCy {#spacy}
The [spaCy](https://spacy.io/) model is trained for several NLP tasks [(read more here)](../frameworks/spacy.md) using the [DDT and DaNE](../datasets.md#dane) annotations.
The spaCy model can be loaded with DaNLP to do NER predictions in the following way.
```python
from danlp.models import load_spacy_model

nlp = load_spacy_model()

doc = nlp('Jens Peter Hansen kommer fra Danmark') 
for tok in doc:
    print("{} {}".format(tok,tok.ent_type_))
```

### Polyglot
The Polyglot [(Al-Rfou et al. 2015)](https://arxiv.org/abs/1410.3791) NER model
is  trained without any human annotation or language-specific knowledge but 
by automatic generating a dataset using the link structure from Wikipedia.
This model is not available through DaNLP but it can be used from the 
[Polyglot](https://github.com/aboSamoor/polyglot) library.


### NERDA

[NERDA](https://github.com/ebanalyse/NERDA/) is a python package 
that provides an interface for fine-tuning pretrained transformers for NER. 
It also includes some ready-to-use fine-tuned (on [DaNE](../datasets.md#dane)) NER models based on a 
[multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md) 
and a [Danish Electra](https://github.com/MalteHB/-l-ctra). 


### DaCy {#dacy}

[DaCy](https://github.com/KennethEnevoldsen/DaCy) is a multi-task transformer trained using SpaCy v.3.
its models is fine-tuned (on [DaNE](../datasets.md#dane)) and based upon the Danish BERT (v2) by [botXO](https://github.com/botxo/nordic_bert) and the [XLM Roberta large](https://huggingface.co/xlm-roberta-large). For more on DaCy see the github [repository](https://github.com/KennethEnevoldsen/DaCy) or the [blog post](https://www.kennethenevoldsen.com/post/new-fast-and-efficient-state-of-the-art-in-danish-nlp/) describing the training procedure. 

### Daner
The daner [(Derczynski et al. 2014)](https://www.aclweb.org/anthology/E14-2016) NER tool
is a wrapper around the [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) 
using data from [(Derczynski et al. 2014)](https://www.aclweb.org/anthology/E14-2016) (not released).
The tool is not available through DaNLP but it can be used from the [daner repository](https://github.com/ITUnlp/daner).

### DaLUKE

The [DaLUKE](https://github.com/peleiden/daluke) model is based on the knowledge-enhanced transformer LUKE. 
It has been first pretrained as a language model on the Danish Wikipedia and then fine-tuned on [DaNE](../datasets.md#dane) for NER. 

## 📈 Benchmarks
The benchmarks has been performed on the test part of the
[DaNE](../datasets.md#dane) dataset.
None of the models have been trained on this test part. We are only reporting the scores on the `LOC`, `ORG` and `PER` entities as the `MISC` category has limited practical use.
The table below has the achieved F1 score on the test set:

| Model                | LOC       | ORG       | PER       | AVG       |
|----------------------|-----------|-----------|-----------|-----------|
| BERT                 | 83.90     | 72.98     | 92.82     | 84.04     |
| Flair                | 84.82     | 62.95     | 93.15     | 81.78     |
| spaCy                | 75.96     | 59.57     | 87.87     | 75.73     |
| Polyglot             | 64.95     | 39.3      | 78.74     | 64.18     |
| NERDA (mBERT)        | 80.75     | 65.73     | 92.66     | 80.66     |
| NERDA (electra)      | 77.67     | 60.13     | 90.16     | 76.77     |
| DaCy (medium) v0.0.0 | 83.96     | 66.23     | 90.41     | 80.09     |
| DaCy (large) v0.0.0  | **85.29** | **79.04** | **95.53** | **86.64** |
| DaLUKE v0.0.5        | 86.43     | 74.58     | 92.52     | 84.91     |

The evaluation script `ner_benchmarks.py` can be found [here](https://github.com/alexandrainst/danlp/blob/master/examples/benchmarks/ner_benchmarks.py).



## 🎓 References
- Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. 2019. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://www.aclweb.org/anthology/N19-1423/). In **NAACL**.
- Rami Al-Rfou, Vivek Kulkarni, Bryan Perozzi and Steven Skiena. 2015. [POLYGLOT-NER: Massive Multilingual Named Entity Recognition](https://arxiv.org/abs/1410.3791). In **SDM**.
- Leon Derczynski, Camilla V. Field and Kenneth S. Bøgh. 2014. [DKIE: Open Source Information Extraction for Danish](https://www.aclweb.org/anthology/E14-2016). In **EACL**.
- Alan Akbik, Duncan Blythe and Roland Vollgraf. 2018. [Contextual String Embeddings for Sequence Labeling](https://www.aclweb.org/anthology/C18-1139/). In **COLING**.
- Rasmus Hvingelby, Amalie B. Pauli, Maria Barrett, Christina Rosted, Lasse M. Lidegaard and Anders Søgaard. 2020. [DaNE: A Named Entity Resource for Danish](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.565.pdf). In **LREC**.
- Kevin Clark, Minh-Thang Luong, Quoc V. Le and Christopher D. Manning. [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://nlp.stanford.edu/pubs/clark2020electra.pdf). In **ICLR**.
- Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda, Yuji Matsumoto. [LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention](https://aclanthology.org/2020.emnlp-main.523/)