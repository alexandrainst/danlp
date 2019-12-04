Sentiment Analysis
============================
This project is working on improving sentiment analysis on Danish. A general example of sentiment analysis is 
to predict if a sentence or text paragraph has a positive or negative tone.

In the litterateur, an often used approach is to take user reviews, 
for example the [IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/), where the data consist of
pairs with a small text paragraph and a rating. Such ratings can be used as a proxy for the sentiment in the text. 

This project is working towards to contribute with such an annotated dataset, and to support the DaNLP package with
a sentiment classifier. 

## **:construction_worker:** Construction of a new dataset  in process
The NoRec project in Norway  have constructed a dataset for sentiment analysis based on reviews from many
Norwegian news organizations, read more about in [Velldal et al. (2017)](http://www.lrec-conf.org/proceedings/lrec2018/pdf/851.pdf).
The DaNLP project hope to create a similar dataset in Danish with permission from the copyright holders.
So if you manage a site containing user reviews for example movie reviews and would like to contribute then please contact us.  

## Available on Danish 
Without any Danish trainings data at the moment, there is a word list approach open source, and then this project provided an example of how to utilize an English dataset to get a classifier that can be used on Danish text. 

##### Wordlist based approach 
[AFINN](https://github.com/fnielsen/afinn/blob/master/LICENSE) is a classifier based on a word list with 
positive and negative words where each word is given a score. This obviously have some shortcomings 
but is a good benchmark. 

##### Cross-lingual transfer example
An example of utilizing an dataset in another language to be able to make predicts on Danish without seeing 
Danish training data is shown in this 
[notebok](<https://github.com/alexandrainst/danlp/blob/sentiment-start/examples/Zero_shot_sentiment_analysi_example.ipynb>).
It uses multilingual embeddings from [Artetxe et al. 2019](https://arxiv.org/pdf/1812.10464.pdf) called 
[LASER](<https://github.com/facebookresearch/LASER>)(Language-Agnostic SEntence Representations).
Note that this is not evaluated on any Danish data, and the performance is therefore unknown however a few examples 
is shown in the notebook.



## 🎓 References 
- Mikel Artetxe, and Holger Schwenk. 2019. 
  [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond.](https://arxiv.org/pdf/1812.10464.pdf). 
  In **TACL**.
- Erik Velldal, Lilja Øvrelid, Eivind Alexander Bergem, Cathrine Stadsnes, Samia Touileb and Fredrik Jørgensen. 2018. 
  [NoReC: The Norwegian Review Corpus.](http://www.lrec-conf.org/proceedings/lrec2018/pdf/851.pdf) In **LREC**.

 
