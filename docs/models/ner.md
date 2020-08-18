Named Entity Recognition
========================
Named Entity Recognition (NER) is the task of extracting named entities in a raw text. 
Common entity types are locations, organizations and persons. Currently a few
tools are available for NER in Danish. Popular models for NER
([BERT](https://huggingface.co/transformers/index.html),
[Flair](https://github.com/flairNLP/flair) and [spaCy](https://spacy.io/))
are continuously trained on the newest available named entity datasets such as DaNE
and made available through the DaNLP library.

| Model | Train Data | Maintainer | Tags | DaNLP |
|-------|-------|------------|------|-------|
| [BERT](https://github.com/alexandrainst/danlp/blob/master/docs/models/ner.md#bert) | [DaNE](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane) | Alexandra Institute | PER, ORG, LOC | ✔ |
| [Flair](https://github.com/alexandrainst/danlp/blob/master/docs/models/ner.md#flair) | [DaNE](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane) | Alexandra Institute | PER, ORG, LOC | ✔️ |
| [spaCy](https://github.com/alexandrainst/danlp/blob/master/docs/models/ner.md#spacy) | [DaNE](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane) | Alexandra Institute | PER, ORG, LOC | ✔ |
| [Polyglot](https://polyglot.readthedocs.io/en/latest/POS.html/#) | Wikipedia | Polyglot | PER, ORG, LOC | ❌ | 
| [daner](https://github.com/ITUnlp/daner) | [Derczynski et al. (2014)](https://www.aclweb.org/anthology/E14-2016) | [ITU NLP](https://nlp.itu.dk/) | PER, ORG, LOC | ❌ |

#### BERT
The BERT [(Devlin et al. 2019)](https://www.aclweb.org/anthology/N19-1423/) NER model is based on the pre-trained [Danish BERT](https://github.com/botxo/nordic_bert) representations by BotXO which 
has been finetuned on the [DaNE](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane) 
dataset [(Hvingelby et al. 2020)](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.565.pdf). The finetuning has been done using the [Transformers](https://github.com/huggingface/transformers) library from HuggingFace.

To use the BERT NER model it can be loaded with the `load_bert_ner_model()` method. 
```python
from danlp.models import load_bert_ner_model

bert = load_bert_ner_model()
tokens, labels = bert.predict("Jens Peter Hansen kommer fra Danmark")

print(" ".join(["{}/{}".format(tok,lbl) for tok,lbl in zip(tokens,labels)]))
```


#### Flair
The Flair [(Akbik et al. 2018)](https://www.aclweb.org/anthology/C18-1139/) NER model
uses pretrained [Flair embeddings](https://github.com/alexandrainst/danlp/blob/master/docs/models/embeddings.md#-training-details-for-flair-embeddings)
in combination with fastText word embeddings. The model is trained using the [Flair](https://github.com/flairNLP/flair)
 library on the the [DaNE](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane) dataset.

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

#### spaCy
The [spaCy](https://spacy.io/) model is trained for several NLP tasks [(read more here)](https://github.com/alexandrainst/danlp/blob/master/docs/spacy.md) uing the [DDT and DaNE](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane) annotations.
The spaCy model can be loaded with DaNLP to do NER predictions in the following way.
```python
from danlp.models import load_spacy_model

nlp = load_spacy_model()

doc = nlp('Jens Peter Hansen kommer fra Danmark') 
for tok in doc:
    print("{} {}".format(tok,tok.ent_type_))
```

#### Polyglot
The Polyglot [(Al-Rfou et al. 2015)](https://arxiv.org/abs/1410.3791) NER model
is  trained without any human annotation or language-specific knowledge but 
by automatic generating a dataset using the link structure from Wikipedia.
This model is not available through DaNLP but it can be used from the 
[Polyglot](https://github.com/aboSamoor/polyglot) library.

#### Daner
The daner [(Derczynski et al. 2014)](https://www.aclweb.org/anthology/E14-2016) NER tool
is a wrapper around the [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) 
using data from [(Derczynski et al. 2014)](https://www.aclweb.org/anthology/E14-2016) (not released).
The tool is not available through DaNLP but it can be used from the [daner repository](https://github.com/ITUnlp/daner).

## 📈 Benchmarks
The benchmarks has been performed on the test part of the
[DaNE](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane) dataset.
None of the models have been trained on this test part. We are only reporting the scores on the `LOC`, `ORG` and `PER` entities as the `MISC` category has limited practical use.
The table below has the achieved F1 score on the test set:

| Model |   LOC | ORG | PER | AVG |
|-------|-------|-----|-----|-----|
| BERT | 83.90 | **72.98** | 92.82 | **84.04** |
| Flair | **84.82** | 62.95 | **93.15** | 81.78 |
| spaCy | 75.96 | 59.57 | 87.87 | 75.73 |
| Polyglot | 64.95 | 39.3 | 78.74 | 64.18 |

The evaluation script `ner_benchmarks.py` can be found [here](https://github.com/alexandrainst/danlp/blob/master/examples/benchmarks/ner_benchmarks.py).



## 🎓 References
- Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. 2019. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://www.aclweb.org/anthology/N19-1423/). In **NAACL**.
- Rami Al-Rfou, Vivek Kulkarni, Bryan Perozzi and Steven Skiena. 2015. [POLYGLOT-NER: Massive Multilingual Named Entity Recognition](https://arxiv.org/abs/1410.3791). In **SDM**.
- Leon Derczynski, Camilla V. Field and Kenneth S. Bøgh. 2014. [DKIE: Open Source Information Extraction for Danish](https://www.aclweb.org/anthology/E14-2016). In **EACL**.
- Alan Akbik, Duncan Blythe and Roland Vollgraf. 2018. [Contextual String Embeddings for Sequence Labeling](https://www.aclweb.org/anthology/C18-1139/). In **COLING**.
- Rasmus Hvingelby, Amalie B. Pauli, Maria Barrett, Christina Rosted, Lasse M. Lidegaard and Anders Søgaard. 2020. [DaNE: A Named Entity Resource for Danish](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.565.pdf). In **LREC**.
