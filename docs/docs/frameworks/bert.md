BERT
====

BERT (Bidirectional Encoder Representations from Transformers) [(Devlin et al. 2019)](https://www.aclweb.org/anthology/N19-1423/) is a deep neural network model used in Natural Language Processing. 

The BERT models provided with DaNLP are based on the pre-trained [Danish BERT](https://github.com/botxo/nordic_bert) representations by BotXO which has been finetuned on several tasks using the [Transformers](https://github.com/huggingface/transformers) library from HuggingFace.

Through DaNLP, we provide fine-tuned BERT models for the following tasks: 

* Named Entity Recognition
* Emotion detection
* Tone and polarity detection

Please note that the BERT models can take a maximum of 512 tokens as input at a time. For longer text sequences, you should split the text before hand -- for example by using sentence boundary detection (e.g. with the [spaCy model](spacy.md)).

See our [getting started guides](../gettingstarted/quickstart.md#bert) for examples on how to use the BERT models. 

### Named Entity Recognition

The BERT NER model has been finetuned on the [DaNE](../datasets.md#dane) dataset [(Hvingelby et al. 2020)](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.565.pdf). 
It can be loaded with the `load_bert_ner_model()` method.  

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

### Tone and polarity detection

The tone analyzer consists of two BERT classification models.
The first model detects the polarity of a sentence, i.e. whether it is perceived as `positive`, `neutral` or `negative`.
The second model detects the tone of a sentence, between `subjective` and `objective`. 

The models are finetuned on manually annotated Twitter data from [Twitter Sentiment](../datasets.md#twitter-sentiment) (train part) and [EuroParl sentiment 2](../datasets.md#europarl-sentiment2)).
Both datasets can be loaded with the DaNLP package.  

