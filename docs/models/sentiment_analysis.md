Sentiment Analysis
============================
This project is working on improving sentiment analysis on Danish.  An general example of sentiment analysis is to predict if a sentence or text paragraf has a poitive or negative tone.

In the litterateur, an often used approach is to take user reviews, for example the [IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/ ), where the data consist of  pairs with a  small text paragraph and  a rating. Such ratings can be used as a proxy for the sentiment in the text. 

This project is working towards to contribute with such an annotated dataset, and to support the DaNLP package with a sentiment classifier. 
As in general in this project, the docs will also focus on what is avalible in Danish, benchmark results and good pratices.  

### Construction of a new dataset  
The NoRec project in Norway  have constructed a dataset for sentiment analysis based on reviews from many  news organizations in Norway, read more about in the [paper](http://www.lrec-conf.org/proceedings/lrec2018/pdf/851.pdf). The DaNLP project hope to create a similar dataset in Danish with permission from the copyright holders. So if you manage a site containing user reviews for example movie reviews and would like to contribute then please contact us.  

### Avalible on Danish 
Without any Danish trainings data at the moment, there is a wordlist approach open source, and then this project provide an example of how to utilize an English dataset to get a classifier that can be used on Danish text. 

##### Wordlist based approach 
[Affin](https://github.com/fnielsen/afinn/blob/master/LICENSE) is open source (Apache License version 2.0). This classifier is based on a wordlist with positive and negative words where each word is given a score. This obviously have some shortcomings but is a good benchmark. 

##### Cross-lingual transfer example

An example of utilizing an dataset in another language to be able to make predicts on Danish without seeing Danish training data is shown in this [notebok](). It is using multilingual embeddings from [LASER](<https://github.com/facebookresearch/LASER>) (Language-Agnostic SEntence Representations). Note that this is not evaluate on any Danish data, and the performance is therefore unknown. But a few examples in the notebook is shown.



