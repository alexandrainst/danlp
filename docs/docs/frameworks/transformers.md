Transformers
============

In DaNLP, we use language representation models based on the Transformer architecure [(Vaswani et al. 2017)](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). 
We build specialized models for the most common NLP tasks by fine-tuning transformer models, such as BERT and XLM-RoBERTa.  


## BERT

BERT (Bidirectional Encoder Representations from Transformers) [(Devlin et al. 2019)](https://www.aclweb.org/anthology/N19-1423/) is a deep neural network model. It is one of the most popular transformer-based model used in Natural Language Processing. 

The BERT models provided with DaNLP are based on the pre-trained [Danish BERT](https://github.com/botxo/nordic_bert) representations by BotXO, and different models have been finetuned on different tasks using the [Transformers](https://github.com/huggingface/transformers) library from HuggingFace.

Through DaNLP, we provide fine-tuned BERT models for the following tasks: 

* Named Entity Recognition
* Emotion detection
* Tone and polarity detection
* Hatespeech detection

The  pre-trained  [Danish BERT](https://github.com/botxo/nordic_bert)  from BotXO can also be used for the following tasks without any further finetuning:

- Embeddings of tokens or sentences 
- Predict a mask word in a sentence 
- Predict  if a sentence naturally follows another sentence

Please note that the BERT models can take a maximum of 512 tokens as input at a time. For longer text sequences, you should split the text before hand -- for example by using sentence boundary detection (e.g. with the [spaCy model](spacy.md)).

### Language model, embeddings and next sentence prediction

The BERT model  [(Devlin et al. 2019)](https://www.aclweb.org/anthology/N19-1423/)  is originally pretrained on two tasks. The first, is to predict a masked word in a sentence, and the second is to predict if a sentence follows another sentence. Therefore, the model can without any further finetuning be used for this two tasks. 

A pytorch version of the  [Danish BERT](https://github.com/botxo/nordic_bert) trained by BotXo can therefore be loaded with the DaNLP package and used through the [Transformers](https://github.com/huggingface/transformers)  library. 

For **predicting a masked word** in a sentence, you can after downloading the model through DaNLP, use the transformer library directly as described in the following snippet:

```python
from transformers import pipeline
from danlp.models import load_bert_base_model
# load the BERT model
model = load_bert_base_model()
# Use the transfomer libary built in function
LM = pipeline("fill-mask", model=model.path_model)
# Use the model as a language model to predict masked words in a sentence
LM(f"Jeg kan godt lide at spise {LM.tokenizer.mask_token}.")  
# output is top five words in a list of dicts
"""
[{'sequence': '[CLS] jeg kan godt lide at spise her. [SEP]',
  'score': 0.15520372986793518,
  'token': 215,
  'token_str': 'her'},
 {'sequence': '[CLS] jeg kan godt lide at spise ude. [SEP]',
  'score': 0.05564282834529877,
  'token': 1500,
  'token_str': 'ude'},
 {'sequence': '[CLS] jeg kan godt lide at spise kød. [SEP]',
  'score': 0.052283965051174164,
  'token': 3000,
  'token_str': 'kød'},
 {'sequence': '[CLS] jeg kan godt lide at spise morgenmad. [SEP]',
  'score': 0.051760803908109665,
  'token': 4538,
  'token_str': 'morgenmad'},
 {'sequence': '[CLS] jeg kan godt lide at spise der. [SEP]',
  'score': 0.049477532505989075,
  'token': 59,
  'token_str': 'der'}]
"""
```

The DaNLP package also provides some wrapper code for **next sentence prediction**:

```python
from danlp.models import load_bert_nextsent_model
model = load_bert_nextsent_model()

# the sentence is from a wikipedia article https://da.wikipedia.org/wiki/Uranus_(planet)
# Sentence B1 follows after sentence A, where sentence B2 is taken futher down in the article
sent_A= "Uranus er den syvende planet fra Solen i Solsystemet og var den første planet der blev opdaget i historisk tid."
sent_B1 =" William Herschel opdagede d. 13. marts 1781 en tåget klat, som han først troede var en fjern komet." 
sent_B2= "Yderligere er magnetfeltets akse 59° forskudt for rotationsaksen og skærer ikke centrum."

# model returns the probability of sentence B follows rigth after sentence A
model.predict_if_next_sent(sent_A, sent_B1)
"""0.9895"""
model.predict_if_next_sent(sent_A, sent_B2)
"""0.0001"""
```

The wrapper function for **embeddings** of tokens or sentences can be read about in the [docs for embeddings](../tasks/embeddings.md).



### Named Entity Recognition

The BERT NER model has been finetuned on the [DaNE](../datasets.md#dane) dataset [(Hvingelby et al. 2020)](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.565.pdf). 
The tagger recognizes the following tags:

- `PER`: person
- `ORG`: organization
- `LOC`: location

Read more about it in the [NER docs](../tasks/ner.md).

### Emotion detection

The emotion classifier is developed in a collaboration with Danmarks Radio, which has granted access to a set of social media data. The data has been manually annotated first to distinguish between a binary problem of emotion or no emotion, and afterwards tagged with 8 emotions. The BERT emotion model is finetuned on this data.

The model can detect the eight following emotions:

* `Glæde/Sindsro`
* `Tillid/Accept`
* `Forventning/Interrese`
* `Overasket/Målløs`
* `Vrede/Irritation`
* `Foragt/Modvilje`
* `Sorg/trist`
* `Frygt/Bekymret`

The model achieves an accuracy of 0.65 and a macro-f1 of 0.64 on the social media test set from DR's Facebook containing 999 examples. We do not have permission to distributing the data. 

Read more about it in the [sentiment docs](../tasks/sentiment_analysis.md).

### Tone and polarity detection

The tone analyzer consists of two BERT classification models.

The models are finetuned on manually annotated Twitter data from [Twitter Sentiment](../datasets.md#twitter-sentiment) (train part) and [EuroParl sentiment 2](../datasets.md#europarl-sentiment2)).
Both datasets can be loaded with the DaNLP package.  

The first model detects the polarity of a sentence, i.e. whether it is perceived as `positive`, `neutral` or `negative`.
The second model detects the tone of a sentence, between `subjective` and `objective`. 

Read more about it in the [sentiment docs](../tasks/sentiment_analysis.md).

### Hatespeech detection

The offensive language identification model predicts whether a text is offensive (`OFF`) or not (`NOT`). 
The model is fine-tuned on the [DKHate](../datasets.md#dkhate) dataset. 

Read more about it in the [hatespeech docs](../tasks/hatespeech.md).

## XLM-RoBERTa

XLM-Roberta is a transformer-based multilingual masked language model [(Conneau et al. 2020)](https://www.aclweb.org/anthology/2020.acl-main.747.pdf). It has shown better performance than mBERT (multilingual BERT) on a range of NLP tasks. 

Through DaNLP, we provide fine-tuned XLM-R models for the tasks of named entity disambiguation and coreference resolution. 

### Named Entity Disambiguation

Named entity disambiguation is a binary classification task which is used to predict whether a specific instance of an entity is mentioned in a text. It is used for named entity linking.

Read more about it in the [NED docs](../tasks/ned.md).

### Coreference resolution

Coreference resolution is the task of finding all expressions that refer to the same entity in a text.

Read more about it in the [coreference docs](../tasks/coreference.md).