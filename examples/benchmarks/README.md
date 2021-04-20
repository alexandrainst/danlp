Benchmarks scripts
==================

This  folder contains scripts to reproduce the benchmark results reported in our [documentation](https://danlp-alexandra.readthedocs.io/en/latest/tasks.html) for the different tasks we cover. 

The benchmark scripts evaluate models implemented in the danlp package but also models implemented in other frameworks.
Therefore more installation of packages is needed to run some of the scripts.
You can either look in the specific scripts to check which packages is needed, or you can install all the packages which is needed by `pip install -r requirements_benchmark.txt`.

For running the `sentiment_benchmarks_twitter` you need a twitter development account and setting the keys as environment variable.

#### List of current benchmark scripts

- Benchmark script for word embeddings in `wordembeddings_benchmarks.py`

- Benchmark script on the
   [DaNE](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank) 
   NER dataset in `ner_benchmarks.py`

- Benchmark script for sentiment classification on [LCC Sentiment](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#lcc-sentiment)  and [Europarl Sentiment](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#europarl-sentiment) using the tools [AFINN](https://github.com/alexandrainst/danlp/blob/master/docs/models/sentiment_analysis.md#afinn) and [Sentida](https://github.com/alexandrainst/danlp/blob/master/docs/models/sentiment_analysis.md#sentida) where the scores are converted to three class problem. It also includes benchmark of [BERT Tone (polarity)](https://github.com/alexandrainst/danlp/blob/master/docs/models/sentiment_analysis.md#wrenchbert-tone)  `sentiment_benchmarks.py`

- `sentiment_benchmarks_twitter.py` show evaluation on a small twitter dataset for both polarity and subjective/objective classification

- Benchmark script of [Part of Speech tagging](<https://github.com/alexandrainst/danlp/blob/master/docs/models/pos.md>) on [Danish Dependency Treebank](<https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane>). SpaCy, Dacy, flair, polyglot and Stanza models are benchmarked `pos_benchmarks.py`

- Benchmark script of [Dependency Parsing](<https://github.com/alexandrainst/danlp/blob/master/docs/models/dependency.md>) on [Danish Dependency Treebank](<https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane>). SpaCy, Dacy and Stanza models are benchmarked `dependency_benchmarks.py`
  
- Benchmark script of Noun-phrase Chunking -- depending on the [Dependency Parsing model](<https://github.com/alexandrainst/danlp/blob/master/docs/models/dependency.md>) -- on [Danish Dependency Treebank](<https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane>). The (convertion of the dependencies given by the) spaCy model is benchmarked `chunking_benchmarks.py`
