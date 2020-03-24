Examples
========

Here you will find examples and tutorials on how to use NLP in Danish.

Some of the examples make use of the DaNLP pip package and show how to use the 
package. Other examples focus on different use case e.g. applying cross-transfer 
learning on Danish. 

The Jupyter notebooks will start with a description of how to setup the 
installation and run it.      


## List of currently available stuff

-  Tutorial of applying zero shot transfer learning to train a sentiment 
   classifier that can be applied on Danish text in a Jupyter notebook:
   `example_zero_shot_sentiment.ipynb`
-  Benchmark scripts for word embeddings in `wordembeddings_benchmarks.py`
-  Benchmark scripts on the
   [DaNE](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank) 
   NER dataset in `ner_benchmarks.py`

- Benchmark script for sentiment classification on [LCC Sentiment](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#lcc-sentiment)  and [Europarl Sentiment](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#europarl-sentiment) using the tools [AFINN](https://github.com/alexandrainst/danlp/blob/master/docs/models/sentiment_analysis.md#afinn) and [Sentida](https://github.com/alexandrainst/danlp/blob/master/docs/models/sentiment_analysis.md#sentida) where the scores are converted to three class problem in `sentiment_benchmarks.py`

