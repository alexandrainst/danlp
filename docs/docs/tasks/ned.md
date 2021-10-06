Named Entity Disambiguation
===========================

Named Entity Disambiguation (NED) is the task of predicting whether a particular entity is mentioned in a sentence. 
For example, given the sentence "Dronning Margrethe er kendt for sin store arkæologiske interesse." , is the entity "Margrethe II of Denmark" (corresponding to the Wikidata QID `Q102139`) mentioned in the sentence? 

NED is the binary version of Named Entity Linking (NEL) which consists in attaching a QID to an entity. 
Given a sentence and a list of QIDs (potentialy mentioned in the text), NED can be used to find the most probable mentioned entity.

NED is usually used in combination with [NER](ner.md).


| Model         | Train Data                    | Maintainer          | Tags  | DaNLP |
|---------------|-------------------------------|---------------------|-------|-------|
| [XLMR](#xlmr) | [DaNED](../datasets.md#daned) | Alexandra Institute | 0 / 1 | ✔     |

### Use cases

One entity name can refer to two different persons/locations/organizations depending on the context. 
Then, in combination with [NER](ner.md), NED can be used for finding entities and their correct reference, and linking them to a description page. 
The result can be used for differentiating between well-known persons and anonymous individuals in order to, for example, semi-anonymize a text. 


## Models

### 🔧 XLMR {#xlmr}

The XLM-R NED model is based on the pre-trained XLM-Roberta, a transformer-based multilingual masked language model [(Conneau et al. 2020)](https://www.aclweb.org/anthology/2020.acl-main.747.pdf), and finetuned on the combination of the [DaWikiNED](../datasets.md#dawikined) dataset and the training part of the [DaNED](../datasets.md#daned) dataset. 
The model has been developed as part of a Master student project (ITU) by Hiêu Trong Lâm and Martin Wu under the supervision of Maria Jung Barrett (ITU) and Ophélie Lacroix (DaNLP).

The XLM-R NED model can be loaded with the `load_xlmr_ned_model()` method. 
(You can also find the model on our [HuggingFace page](https://huggingface.co/DaNLP/da-bert-ned).)
Please note that the model take maximum 512 tokens as input at a time. Longer text sequences will be truncated.


```python
from danlp.models import load_xlmr_ned_model

xlmr = load_xlmr_ned_model()

sentence = "Karen Blixen vendte tilbage til Danmark, hvor hun boede resten af sit liv på Rungstedlund, som hun arvede efter sin mor i 1939"

# to check if the entity "Karen Blixen" correspond to the QID Q182804
# you have to first generate the knowledge graph (KG) context of this QID
# use the get_kg_context_from_wikidata_qid function from danlp.utils to get the KG
# or the DaNED or DaWikiNED dataset to get the corresponding KG string (see doc below)
# (the following example has been truncated -- use the full KG)
kg_context = "udmærkelser modtaget Kritikerprisen udmærkelser modtaget Tagea Brandts Rejselegat udmærkelser modtaget Ingenio ..."

label = xlmr.predict(sentence, kg_context)
```

For more details about how to generate a KG context adapted to the model, see the [DaNED](../datasets.md#daned) and [DaWikiNED](../datasets.md#dawikined) documentation.



## 📈 Benchmarks
The benchmarks has been performed on the test part of the [DaNED](../datasets.md#daned) dataset.
None of the models have been trained on this test part. 
See F1 scores below (QID is mentioned == label 1 -- QID is not mentioned == label 0) :

| Model | QID is mentioned | QID is not mentioned | AVG   | Sentences per second (CPU*) |
|-------|------------------|----------------------|-------|-----------------------------|
| XLMR  | 83.77            | 91.02                | 87.40 | ~1                          |

*Sentences per second is based on a Macbook Pro with Apple M1 chip.

The evaluation script `ned_benchmarks.py` can be found [here](https://github.com/alexandrainst/danlp/blob/master/examples/benchmarks/ned_benchmarks.py).


## 🎓 References
- Maria Jung Barrett, Hiêu Trong Lâm, Martin Wu, Ophélie Lacroix, Barbara Plank and Anders Søgaard. Resources and Evaluations for Danish Entity Resolution. In **CRAC 2021**.