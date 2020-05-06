Sentiment Analysis
============================

Sentiment analysis refers to identifying an emotion or opinion in a text.
The following focuses on the polarity of a sentence i.e. the tone of positive, neutral or negative.
In this repository we provide an overview of available sentiment analysis models and dataset for Danish. 

In Danish there is so-far no open source annotated training set. 
Two sentiment analysis tools currently exist in Danish. 

| Model                                              | Type     | License                    | Trained by               | Tags                                                         |
| -------------------------------------------------- | -------- | -------------------------- | ------------------------ | ------------------------------------------------------------ |
| [AFINN](https://github.com/alexandrainst/danlp/blob/master/docs/models/sentiment_analysis.md#afinn) | Wordlist | [Apache 2.0](https://github.com/fnielsen/afinn/blob/master/LICENSE) | Finn Årup Nielsen | Score (integers) |
| [Sentida](https://github.com/alexandrainst/danlp/blob/master/docs/models/sentiment_analysis.md#sentida) | Wordlist | [GPL-3.0](https://github.com/esbenkc/emma/blob/master/LICENSE) | Jacob Dalsgaard, Lars Kjartan Svenden og Gustav Lauridsen | Score (continuous) |

#### AFINN
The [AFINN](https://github.com/fnielsen/afinn) tool [(Nielsen 2011)](https://arxiv.org/abs/1103.2903) uses a lexicon based approach for sentiment analysis.
The tool scores texts with an integer where scores <0 are negative, =0 are neutral and >0 are positive. 


#### Sentida
The tool Sentida  [(Lauridsen et al. 2019)](https://tidsskrift.dk/lwo/article/view/115711)
uses a lexicon based approach to sentiment analysis. The tool scores texts with a continuous value. There exist to versions of the tool where the second version is an implementation in Python:  [Sentida](https://github.com/esbenkc/emma) and in these documentations we evaluate this  second version. 


## 📈 Benchmarks 
The benchmark is made by converting the relevant models scores and relevant datasets scores 
into the there classes 'positive', 'neutral' and 'negative'.

The tools are benchmarked on the following datasets:

- [LCC Sentiment](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#lcc-sentiment) contains 499 sentences from the proceedings of the European Parliament annotated with a sentiment score from -5 to 5 by Finn Årup Nielsen.
- [Europarl Sentiment](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#europarl-sentiment) contains 184 sentences from news and web pages annotated with sentiment -5 to 5 by Finn Årup Nielsen.

A conversion of the scores of the LCC and Europarl Sentiment dataset and the Afinn model is done in the following way: a score of zero to be "neutral", a positive score to be "positive" and a negative score to be "negative". 

An conversion of the continuous scores of the Sentida tool into three classes is not given since the 'neutral' class  can not be assumed to be only exactly zero but instead we assume it to be an area around zero.  We looked for a threshold to see how closed to zero a score should be to be interpreted as neutral.   A symmetric threshold is found by optimizing the macro-f1 score on a twitter sentiment corpus (with 1327 examples (the corpus is under construction and will be released later on)) . The threshold is found to be 0.4, which makes our chosen conversion to be:  scores over 0.4 to be 'positive', under -0.4 to be 'negative'  and scores between to be neutral. 

The script for the benchmarks can be found [here](https://github.com/alexandrainst/danlp/blob/master/examples/benchmarks/sentiment_benchmark.py).
In the table we consider the accuracy and macro-f1 in brackets, but to get the scores per class we refer to our benchmark script.

| Tool | Europarl Sentiment | LCC Sentiment |
| ---- | ------------------ | ------------- |
| AFINN | **0.68** (0.68) | **0.66** (0.61) |
| Sentida | 0.67 (0.65) | 0.58 (0.55) |

## **👷** Construction of a new dataset  in process
This project is in the process of creating a manual annotating dataset to use in training of new sentiment analysis models. The dataset will be open source and is therefore using data from varied source such as twitter and [europarl data](<http://www.statmt.org/europarl/>). 

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
