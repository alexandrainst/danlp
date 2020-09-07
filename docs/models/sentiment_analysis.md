Sentiment Analysis
============================

Sentiment analysis is a broad term for a set of tasks with the purpose of identifying an emotion or opinion in a text.

In this repository we provide an overview of open sentiment analysis models and dataset for Danish. 

| Model                                                        | Model    | License                                                      | Trained by                                                | Dimension          | Tags                                                         | DaNLP |
| ------------------------------------------------------------ | -------- | ------------------------------------------------------------ | --------------------------------------------------------- | ------------------ | ------------------------------------------------------------ | ----- |
| [AFINN](https://github.com/alexandrainst/danlp/blob/master/docs/models/sentiment_analysis.md#afinn) | Wordlist | [Apache 2.0](https://github.com/fnielsen/afinn/blob/master/LICENSE) | Finn Årup Nielsen                                         | Polarity           | Score (integers)                                             | ❌     |
| [Sentida](https://github.com/alexandrainst/danlp/blob/master/docs/models/sentiment_analysis.md#sentida) | Wordlist | [GPL-3.0](https://github.com/esbenkc/emma/blob/master/LICENSE) | Jacob Dalsgaard, Lars Kjartan Svenden og Gustav Lauridsen | Polarity           | Score (continuous)                                           | ❌     |
| [Bert Emotion](https://github.com/alexandrainst/danlp/blob/master/docs/models/sentiment_analysis.md#wrenchbert-emotion) | BERT     | CC-BY_4.0                                                    | Alexandra Institute                                       | Emotions           | glæde/sindsro, forventning/interesse, tillid/accept,  overraskelse/forundring, vrede/irritation, foragt/modvilje, sorg/skuffelse, frygt/bekymring, No emotion | ✔️     |
| [Bert Tone](https://github.com/alexandrainst/danlp/blob/master/docs/models/sentiment_analysis.md#wrenchbert-tone) (beta) | BERT     | CC-BY_4.0                                                    | Alexandra Institute                                       | Polarity, Analytic | ['postive', 'neutral', 'negative'] and ['subjective', 'objective] | ✔️     |
| [SpaCy Sentiment](https://github.com/alexandrainst/danlp/blob/master/docs/models/sentiment_analysis.md#wrench-spacy-sentiment) (beta) | spaCy    | MIT                                                          | Alexandra Institute                                       | Polarity           | 'postive', 'neutral', 'negative'                             | ✔️     |



#### AFINN

The [AFINN](https://github.com/fnielsen/afinn) tool [(Nielsen 2011)](https://arxiv.org/abs/1103.2903) uses a lexicon based approach for sentiment analysis.
The tool scores texts with an integer where scores <0 are negative, =0 are neutral and >0 are positive. 


#### Sentida
The tool Sentida  [(Lauridsen et al. 2019)](https://tidsskrift.dk/lwo/article/view/115711)
uses a lexicon based approach to sentiment analysis. The tool scores texts with a continuous value. There exist two versions of the tool where the second version is an implementation in Python:  [Sentida](https://github.com/esbenkc/emma) and in these documentations we evaluate this second version. 

#### :wrench:Bert Emotion

The emotion classifier is developed in a collaboration with Danmarks Radio, which has granted access to a set of social media data. The data has been manual annotated first to distinguish between a binary problem of emotion or no emotion, and afterwards tagged with 8 emotions. The BERT  [(Devlin et al. 2019)](https://www.aclweb.org/anthology/N19-1423/) emotion model is finetuned on this data using the [Transformers](https://github.com/huggingface/transformers) library from HuggingFace, and it is based on a pretrained  [Danish BERT](https://github.com/botxo/nordic_bert) representations by BotXO . The model to classify the eight emotions achieves an accuracy on 0.65 and a macro-f1 on 0.64 on the social media test set from DR's Facebook containing 999 examples. We do not have permission to distributing the data. 

 Below is a small snippet for getting started using the Bert Emotion model. Please notice that the BERT model can maximum take 512 tokens as input, however the code allows for overfloating tokens and will therefore not give an error but just a warning. 

```python
from danlp.models import load_bert_emotion_model
classifier = load_bert_emotion_model()

# using the classifier
classifier.predict('bilen er flot')
''''No emotion''''
classifier.predict('jeg ejer en rød bil og det er en god bil')
''''Tillid/Accept''''
classifier.predict('jeg ejer en rød bil men den er gået i stykker')
''''Sorg/trist''''

# Get probabilities and matching clases names
classifier.predict_proba('jeg ejer en rød bil men den er gået i stykker', no_emotion=False)
classifier._classes()
```



#### :wrench:Bert Tone

The tone analyzer consists of two BERT  [(Devlin et al. 2019)](https://www.aclweb.org/anthology/N19-1423/)  classification models, and the first is recognizing the following tags positive, neutral and negative and the  second model  the tags: subjective and objective. This is a first version of the models, and work should be done to improve performance. Both models is finetuned on annotated twitter data using the [Transformers](https://github.com/huggingface/transformers) library from HuggingFace, and it is based on a pretrained  [Danish BERT](https://github.com/botxo/nordic_bert) representations by BotXO .  The data used is manually annotated data from Twitter Sentiment (train part)([see here](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#twitter-sentiment) ) and EuroParl sentiment 2 ([se here](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#europarl-sentiment2)), both datasets can be loaded with the DaNLP package.  

 Below is a small snippet for getting started using the Bert Tone model. Please notice that the BERT model can maximum take 512 tokens as input, however the code allows for overfloating tokens and will therefore not give an error but just a warning. 

```python
from danlp.models import load_bert_tone_model
classifier = load_bert_tone_model()

# using the classifier
classifier.predict('Analysen viser, at økonomien bliver forfærdelig dårlig')
'''{'analytic': 'objektive', 'polarity': 'negative'}''' 
classifier.predict('Jeg tror alligvel, det bliver godt')
'''{'analytic': 'subjektive', 'polarity': 'positive'}'''

# Get probabilities and matching clases names
classifier.predict_proba('Analysen viser, at økonomien bliver forfærdelig dårlig')
classifier._clases()
```



#### :wrench: SpaCy Sentiment

SpaCy sentiment is a text classification model trained using spacy built in command line interface. It uses the CoNLL2017 word vectors, read about it [here](https://github.com/alexandrainst/danlp/blob/master/docs/models/embeddings.md) .

The model is trained using hard distil of the [Bert Tone](https://github.com/alexandrainst/danlp/blob/master/docs/models/sentiment_analysis.md#wrenchbert-tone) (beta) - Meaning,  the Bert Tone model is used to make predictions on 50.000 sentences from Twitter and 50.000 sentences from [Europarl7](http://www.statmt.org/europarl/). These data is then used to trained a spacy model. Notice the dataset has first been balanced between the classes by oversampling. The model recognizes the classses: 'positiv', 'neutral' and 'negative'.

It is a first version. 

Read more about using the Danish spaCy model [here](https://github.com/alexandrainst/danlp/blob/Add_spacy_sentiment/docs/spacy.md)  

Below is a small snippet for getting started using the spaCy sentiment model. Currently the danlp packages provide both a spaCy model which do not provide any classes in the textcat module (so it is empty for you to train from scratch), and the sentiment spacy model which have pretrained the classes 'positiv', 'neutral' and 'negative'. Notice it is possible with the spacy command line interface to continue training of the sentiment classes, or add new tags. 

```python
from danlp.models import load_spacy_model
# load the model
nlp = load_spacy_model(textcat='sentiment') # if you got an error saying da.vectors not found, try setting vectorError=True - it is an temp fix

# use the model for prediction
doc = nlp("Vi er glade for spacy!")
max(doc.cats.items(), key=operator.itemgetter(1))[0]
'''positiv'''
```






## 📈 Benchmarks  
##### Benchmark of polarity classification

The benchmark is made by converting the relevant models scores and relevant datasets scores 
into the there classes 'positive', 'neutral' and 'negative'.

The tools are benchmarked on the following datasets:

- [LCC Sentiment](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#lcc-sentiment) contains 499 sentences from the proceedings of the European Parliament annotated with a sentiment score from -5 to 5 by Finn Årup Nielsen.
- [Europarl Sentiment](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#europarl-sentiment) contains 184 sentences from news and web pages annotated with sentiment -5 to 5 by Finn Årup Nielsen.
- [Twitter Sentiment](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#twitter-sentiment) contains annotations for polarity (positive, neutral, negative) and annotations for analytic (subjective, objective) made by Alexandra Institute. 512 examples of the dataset are defined for evaluation. 

A conversion of the scores of the LCC and Europarl Sentiment dataset and the Afinn model is done in the following way: a score of zero to be "neutral", a positive score to be "positive" and a negative score to be "negative". 

A conversion of the continuous scores of the Sentida tool into three classes is not given since the 'neutral' class  can not be assumed to be only exactly zero but instead we assume it to be an area around zero.  We looked for a threshold to see how closed to zero a score should be to be interpreted as neutral.   A symmetric threshold is found by optimizing the macro-f1 score on a twitter sentiment corpus (with 1327 examples (the corpus is under construction and will be released later on)) . The threshold is found to be 0.4, which makes our chosen conversion to be:  scores over 0.4 to be 'positive', under -0.4 to be 'negative' and scores between to be neutral. 

The scripts for the benchmarks can be found [here](https://github.com/alexandrainst/danlp/blob/master/examples/benchmarks/). There is one for the europarl sentiment and LCC sentiment data and another one for the twitter sentiment. This is due to the fact that downloading the twitter data requires login to a twitter API account. The scores below for the twitter data is reported for all the data, but if tweets are delete in the mean time on twitter, not all tweets can be downloaded. 
In the table we consider the accuracy and macro-f1 in brackets, but to get the scores per class we refer to our benchmark script.

| Tool/Model | Europarl Sentiment | LCC Sentiment | Twitter Sentiment (Polarity) |
| ---- | ------------------ | ------------- | ---- |
| AFINN | 0.68 (0.68) | 0.66 (0.61) | 0.48 (0.46) |
| Sentida | 0.67 (0.65) | 0.58 (0.55) | 0.44 (0.44) |
| Bert Tone (polarity, version 0.0.1) | **0.79** (0.78) | **0.74** (0.67) | **0.73** (0.70) |
| spaCy sentiment (version 0.0.1) | 0.74 (0.73) | 0.66 (0.61) | 0.66 (0.60) |

**Benchmark of subjective versus objective classification**

The data for benchmark is: 

- [Twitter Sentiment](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#twitter-sentiment) contains annotations for polarity (positive, neutral, negative) and annotations for analytic (subjective, objective) made by Alexandra Institute. 512 examples of the dataset are defined for evaluation. 

The script for the benchmarks can be found [here](https://github.com/alexandrainst/danlp/blob/master/examples/benchmarks/) and it provides more detailed scores. Below is accuracy and macro-f1 reported:

| Model                               | Twitter sentiment (analytic) |
| ----------------------------------- | ---------------------------- |
| Bert Tone (analytic, version 0.0.1) | 0.90 (0.77)                  |



## **👷** Construction of a new dataset  in process

This project is in the process of creating a manual annotating dataset to use in training of new sentiment analysis models. The dataset will be open source and therefore it is using data from varied source such as twitter and [europarl data](<http://www.statmt.org/europarl/>). 

Another approach is to use a "silver" annotated corpus e.g. by using user-reviews and ratings. An example os this is The NoReC corpus [(Velldal et al. 2018)](http://www.lrec-conf.org/proceedings/lrec2018/pdf/851.pdf)  which is a dataset for sentiment analysis in Norwegian based on reviews from many Norwegian news organizations.
The DaNLP project hope to create a similar dataset in Danish with permission from the copyright holders.
So if you manage a site containing user reviews for example movie reviews and would like to contribute then please contact us.

## Zero-shot Cross-lingual transfer example

An example of utilizing an dataset in another language to be able to make predicts on Danish without seeing Danish training data is shown in this 
[notebok](<https://github.com/alexandrainst/danlp/blob/sentiment-start/examples/Zero_shot_sentiment_analysi_example.ipynb>). It is trained on English movie reviews from IMDB, and
it uses multilingual embeddings from [Artetxe et al. 2019](https://arxiv.org/pdf/1812.10464.pdf) called 
[LASER](<https://github.com/facebookresearch/LASER>)(Language-Agnostic SEntence Representations).


## 🎓 References 
- Mikel Artetxe, and Holger Schwenk. 2019. 
  [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond.](https://arxiv.org/pdf/1812.10464.pdf). 
  In **TACL**.
- Erik Velldal, Lilja Øvrelid, Eivind Alexander Bergem, Cathrine Stadsnes, Samia Touileb and Fredrik Jørgensen. 2018. [NoReC: The Norwegian Review Corpus.](http://www.lrec-conf.org/proceedings/lrec2018/pdf/851.pdf) In **LREC**.
- Gustav Aarup Lauridsen, Jacob Aarup Dalsgaard and Lars Kjartan Bacher Svendsen. 2019. [SENTIDA: A New Tool for Sentiment Analysis in Danish](https://tidsskrift.dk/lwo/article/view/115711). In **Sprogvidenskabeligt Studentertidsskrift**.
- Finn Årup Nielsen. 2011. [A new ANEW: evaluation of a word list for sentiment analysis in microblogs](https://arxiv.org/abs/1103.2903). In **CEUR Workshop Proceedings**.
