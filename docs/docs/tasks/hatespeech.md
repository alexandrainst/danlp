Hate Speech Detection
=====================

Hate speech detection is a general term that can include several different tasks. 
The most common is the identification of offensive language which aims at detecting whether a text is offensive or not (e.g. any type of comment that should be moderated on a social media platform such as containing bad language or attacking an individual). 
Once a text is detected as offensive, one can detect whether the content is hateful or not. 

Here are definitions of the previous concepts: 

 * offensive : contains profanity or insult
 * hateful : targets a group or an individual with the intent to be harmful or to cause social chaos.
 

| Model                         | Train Data                      | License         | Trained by          | Tags      | DaNLP |
|-------------------------------|---------------------------------|-----------------|---------------------|-----------|-------|
| [BERT Offensive](#bertdk)     | [DKHate](../datasets.md#dkhate) | CC BY 4.0       | Alexandra Instittut | OFF / NOT | ✔️    |
| [A&ttack](#attack)            | Facebook comments               | CC BY-NC-SA 4.0 | Analyse & Tal       | OFF / NOT | ❌     |
| [BERT Hatespeech](#bertdr)    | Facebook comments               | CC BY 4.0       | Alexandra Instittut | OFF / NOT | ✔️    |
| [ELECTRA Offensive](#electra) | Facebook comments               | CC BY 4.0       | Alexandra Instittut | OFF / NOT | ✔️    |

### Use cases 

Hate speech detection is mostly used with the aim of providing support to moderators of social media platform. 

## Models

### 🔧 BERT Offensive {#bertdk}

The offensive language identification model is intended to solve the binary classification problem of identifying whether a text is offensive or not (contains profanity or insult), therefore, given a text, can predict two classes: `OFF` (offensive) or `NOT` (not offensive). 
Its architecture is based on BERT [(Devlin et al. 2019)](https://www.aclweb.org/anthology/N19-1423/). 
In particular, it is based on the pretrained [Danish BERT](https://github.com/botxo/nordic_bert) trained by BotXO and finetuned on the [DKHate](../datasets.md#dkhate) data using the [Transformers](https://github.com/huggingface/transformers) library. 

The BERT Offensive model can be loaded with the `load_bert_offensive_model()` method. 
Please note that it can maximum take 512 tokens as input at a time. The sentences are automatically truncated if longer.

Below is a small snippet for getting started using the BERT Offensive model. 

```python
from danlp.models import load_bert_offensive_model

# load the offensive language identification model
offensive_model = load_bert_offensive_model()

sentence = "Han ejer ikke respekt for nogen eller noget... han er megaloman og psykopat"

# apply the model on the sentence to get the class in which it belongs
pred = offensive_model.predict(sentence)
# or to get its probability of being part of each class
proba = offensive_model.predict_proba(sentence)
```

### 🔧 BERT HateSpeech {#bertdr}

The BERT HateSpeech model can detect offensive language and hate speech. 

It has been developed in collaboration with Danmarks Radio (DR). 
It is based on the pre-trained [Danish BERT](https://github.com/botxo/nordic_bert) trained by BotXO, and finetuned on facebook data (non publicly available) annotated by DR. 

It can predict:

* whether a text is offensive or not : `OFF` (offensive) or `NOT` (not offensive);
* the hate speech category it falls in : `Særlig opmærksomhed`, `Personangreb`, `Sprogbrug`, `Spam & indhold`.

The BERT HateSpeech model can be loaded with the `load_bert_hatespeech_model()` method. 
Please note that it can maximum take 512 tokens as input at a time. The sentences are automatically truncated if longer.

Below is a small snippet for getting started using the BERT HateSpeech model. 

```python
from danlp.models import load_bert_hatespeech_model

# load the HateSpeech model
hatespeech_model = load_bert_hatespeech_model()

sentence = "Han ejer ikke respekt for nogen eller noget... han er megaloman og psykopat"

# apply the model on the sentence to get the class in which it belongs
pred = hatespeech_model.predict(sentence)
# or to get its probability of being part of each class
proba = hatespeech_model.predict_proba(sentence)
```

### 🔧 ELECTRA Offensive {#electra}

The ELECTRA Offensive model can detect offensive language. 

It has been developed in collaboration with Danmarks Radio (DR). 
It is based on the pre-trained [Danish Ælæctra](Maltehb/aelaectra-danish-electra-small-cased), and finetuned on facebook data (non publicly available) annotated by DR. 

It can predict whether a text is offensive or not : `OFF` (offensive) or `NOT` (not offensive).

The ELECTRA Offensive model can be loaded with the `load_electra_offensive_model()` method. 
Please note that it can maximum take 512 tokens as input at a time. The sentences are automatically truncated if longer.

Below is a small snippet for getting started using the ELECTRA Offensive model. 

```python
from danlp.models import load_electra_offensive_model

# load the model
offensive_model = load_electra_offensive_model()

sentence = "Han ejer ikke respekt for nogen eller noget... han er megaloman og psykopat"

# apply the model on the sentence to get the class in which it belongs
pred = offensive_model.predict(sentence)
# or to get its probability of being part of each class
proba = offensive_model.predict_proba(sentence)
```

### A&ttack (Analyse & Tal) {#attack}

The A&ttack model detects whether a text is offensive or not. It has been developed by [Analyse & Tal](https://ogtal.dk/) and is based on the pretrained [Ælectra model](https://huggingface.co/Maltehb/-l-ctra-danish-electra-small-uncased). It has been trained on social media data (Facebook, 67,188 tokens). 
See the [github repo](https://github.com/ogtal/A-ttack) for more details and the [report](https://strapi.ogtal.dk/uploads/966f1ebcfa9942d3aef338e9920611f4.pdf) of the project.


## 📈 Benchmarks

See detailed scoring of the benchmarks in the [example](<https://github.com/alexandrainst/danlp/tree/master/examples>) folder.

The benchmarking has been performed on the test part of the [DKHate](../datasets.md#dkhate) dataset.

The scores presented here describe the performance (F1) of the models for the task of offensive language identification. 

| Model             | OFF  | NOT  | AVG F1 | Sentences per second (CPU*) |
|-------------------|------|------|--------|-----------------------------|
| BERT Offensive    | 61.9 | 95.4 | 78.7   | ~6                          |
| A&ttack (A&T)     | 34.2 | 91.4 | 62.8   | ~2                          |
| BERT HateSpeech   | 46.2 | 92.8 | 69.5   | ~10                         |
| ELECTRA Offensive | 48.6 | 93.9 | 71.2   | ~60                         |

*Sentences per second is based on a Macbook Pro with Apple M1 chip.

The evaluation script `hatespeech_benchmarks.py` can be found [here](https://github.com/alexandrainst/danlp/blob/master/examples/benchmarks/hatespeech_benchmarks.py).


## 🎓 References 

- Marc Pàmies, Emily Öhman, Kaisla Kajava, Jörg Tiedemann. 2020. [LT@Helsinki at SemEval-2020 Task 12: Multilingual or Language-specific BERT?](https://aclanthology.org/2020.semeval-1.205/). In **SemEval-2020**

  
