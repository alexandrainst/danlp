Coreference Resolution
======================

Coreference resolution is the task of finding all expressions that refer to the same entity in a text.

Typically, in a document, persons are first introduced by their name (e.g. `Dronning Margrethe II`) and later refered by pronouns (e.g. `hun`) or expressions (e.g. `Hendes Majestæt`, `Danmarks dronning`, etc). 
The goal of the coreference resolution task is to find all these references and link them through a common ID. 


| Model           | Train Data                                        | License | Trained by          | Tags            | DaNLP |
|-----------------|---------------------------------------------------|---------|---------------------|-----------------|--|
| [XLM-R](#xlmr)  | [Dacoref](../datasets.md#dacoref)                 | GPLv2   | Maria Jung Barrett  | Generic QIDs    | ✔️     |


#### 🔧 XLM-R {#xlmr}

The XLM-R Coref model is based on the pre-trained XLM-Roberta, a transformer-based multilingual masked language model [(Conneau et al. 2020)](https://www.aclweb.org/anthology/2020.acl-main.747.pdf), and finetuned on the [Dacoref](../datasets.md#dacoref)
dataset. 
The finetuning has been done using the pytorch-based implementation from [AllenNLP 1.3.0.](https://github.com/allenai/allennlp).

The XLM-R Coref model can be loaded with the `load_xlmr_coref_model()` method. 
Please note that it can maximum take 512 tokens as input at a time. For longer text sequences split before hand, for example using sentence boundary detection (e.g. by using the [spacy model](../frameworks/spacy.md ).) 

```python
from danlp.models import load_xlmr_coref_model

# load the coreference model
coref_model = load_xlmr_coref_model()

# a document is a list of tokenized sentences
doc = [["Lotte", "arbejder", "med", "Mads", "."], ["Hun", "er", "tandlæge", "."]]

# apply coreference resolution on the document
preds = xlmr_model.predict(doc)
```

## 📈 Benchmarks

See detailed scoring of the benchmarks in the [example](<https://github.com/alexandrainst/danlp/tree/master/examples>) folder.

The benchmarks has been performed on the test part of the [Dacoref](../datasets.md#dacoref) dataset.


| Model | Precision | Recall | F1    | Mention Recall |
|-------|-----------|--------|-------|----------------|
| XLM-R  | 69.86     | 59.17  | 64.02 | 88.01          |


The evaluation script `coreference_benchmarks.py` can be found [here](https://github.com/alexandrainst/danlp/blob/master/examples/benchmarks/coreference_benchmarks.py).
