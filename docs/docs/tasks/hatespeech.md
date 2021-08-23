Hate Speech Detection
=====================

Hate speech detection is a general term that can include several different tasks. 
The most common is the identification of offensive language which aims at detecting whether a text is offensive or not (e.g. any type of comment that should be moderated on a social media platform such as containing bad language or attacking an individual). 
Once a text is detected as offensive, one can detect whether the content is hateful or not. 

Here are definitions of the previous concepts: 
 * offensive : contains profanity or insult
 * hateful : targets a group or an individual with the intent to be harmful or to cause social chaos.
 

| Model         | Train Data                      | License   | Trained by          | Tags      | DaNLP |
|---------------|---------------------------------|-----------|---------------------|-----------|-------|
| [BERT](#bert) | [DKHate](../datasets.md#dkhate) | CC BY 4.0 | Alexandra Instittut | OFF / NOT | ✔️    |


### Use cases 

Hate speech detection is mostly used with the aim of providing support to moderators of social media platform. 

## Models

### 🔧 BERT Offensive {#bert}

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


## 📈 Benchmarks

See detailed scoring of the benchmarks in the [example](<https://github.com/alexandrainst/danlp/tree/master/examples>) folder.

The benchmarks has been performed on the test part of the [DKHate](../datasets.md#dkhate) dataset.

The scores presented here describe the performance of the models for the task of offensive language identification. 

| Model | OFF  | NOT  | AVG F1 |
|-------|------|------|--------|
| BERT  | 61.9 | 95.4 | 78.7   |


The evaluation script `hatespeech_benchmarks.py` can be found [here](https://github.com/alexandrainst/danlp/blob/master/examples/benchmarks/hatespeech_benchmarks.py).


## 🎓 References 

- Marc Pàmies, Emily Öhman, Kaisla Kajava, Jörg Tiedemann. 2020. [LT@Helsinki at SemEval-2020 Task 12: Multilingual or Language-specific BERT?](https://aclanthology.org/2020.semeval-1.205/). In **SemEval-2020**

  
