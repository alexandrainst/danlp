Coreference Resolution
======================

Coreference resolution is the task of finding all mentions (noun phrases) that refer to the same entity (e.g. a person, a location etc, see also the [NER](#ner) doc) in a text.

Typically, in a document, entities are first introduced by their name (e.g. `Dronning Margrethe II`) and later refered by pronouns (e.g. `hun`) or expressions/titles (e.g. `Hendes Majestæt`, `Danmarks dronning`, etc). 
The goal of the coreference resolution task is to find all these references and link them through a common ID. 


| Model           | Train Data                                        | License | Trained by          | Tags            | DaNLP |
|-----------------|---------------------------------------------------|---------|---------------------|-----------------|--|
| [XLM-R](#xlmr)  | [Dacoref](../datasets.md#dacoref)                 | GPLv2   | Maria Jung Barrett  | Generic QIDs    | ✔️     |


If you want to read more about coreference resolution and the DaNLP model, we also have a [blog post](https://medium.com/danlp/coreferensmodeller-nu-ogs%C3%A5-p%C3%A5-dansk-5aea04f4876e) (in Danish).


### Use cases 

Coreference resolution is an important subtask in NLP. It is used in particular for information extraction (e.g. for building a knowledge graph, see our [tutorial](https://github.com/alexandrainst/danlp/blob/master/examples/tutorials/example_knowledge_graph.ipynb)) and could help with other NLP tasks such as machine translation (e.g. in order to apply the right gender or number) or text summarization, or in dialog systems. 

## Models

### 🔧 XLM-R {#xlmr}

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

# apply coreference resolution to the document and get a list of features (see below)
preds = coref_model.predict(doc)

# apply coreference resolution to the document and get a list of clusters
clusters = coref_model.predict_clusters(doc)
```

The `preds` variable is a dictionary including the following entries :
* `top_spans` : list of indices of all references (spans) in the document
* `antecedent_indices` : list of antecedents indices
* `predicted_antecedents` : list of indices of the antecedent span (from `top_spans`), i.e. previous reference
* `document` : list of tokens' indices for the whole document
* `clusters` : list of clusters (indices of tokens)
The most relevant entry to use is the list of clusters. One cluster contains the indices of references (spans) that refer to the same entity.
To make it easier, we provide the `predict_clusters` function that returns a list of the clusters with the references and their ids in the document.


## 📈 Benchmarks

See detailed scoring of the benchmarks in the [example](<https://github.com/alexandrainst/danlp/tree/master/examples>) folder.

The benchmarks has been performed on the test part of the [Dacoref](../datasets.md#dacoref) dataset.


| Model | Precision | Recall | F1    | Mention Recall |
|-------|-----------|--------|-------|----------------|
| XLM-R  | 69.86     | 59.17  | 64.02 | 88.01          |


The evaluation script `coreference_benchmarks.py` can be found [here](https://github.com/alexandrainst/danlp/blob/master/examples/benchmarks/coreference_benchmarks.py).
