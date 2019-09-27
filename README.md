<h1 align="center">
  <img src="https://raw.githubusercontent.com/alexandrainst/danlp/master/docs/imgs/danlp_logo.png"  width="350"  />
</h1>
 
<div align="center">
  <a href="https://pypi.org/project/danlp/"><img src="https://img.shields.io/pypi/v/danlp.svg"></a>
  <a href="https://travis-ci.org/alexandrainst/danlp"><img src="https://travis-ci.org/alexandrainst/danlp.svg?branch=master"></a>
  <a href="https://coveralls.io/github/alexandrainst/danlp?branch=master"><img src="https://coveralls.io/repos/github/alexandrainst/danlp/badge.svg?branch=master"></a>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/license-BSD%203--Clause-blue.svg"></a>
</div>

<div align="center">
  <h3>
    <a href="https://github.com/alexandrainst/danlp/tree/master/docs/models">
      Models
    </a>
    <span> | </span>
    <a href="https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md">
      Datasets
    </a>
    <span> | </span>
    <a href="https://github.com/alexandrainst/danlp/tree/master/examples">
      Examples
    </a>
  </h3>
</div>

DaNLP is a repository for Natural Language Processing resources for the Danish Language. 
It is a collection  of available datasets and models for a a variety of NLP tasks.
It features code examples on how to use the datasets and models in popular NLP frameworks such as spaCy and NLTK as 
well as Deep Learning frameworks such as PyTorch and TensorFlow.

**News**

- ✨ The Danish Dependency Treebank has been annotated with NER tags. You can use DaNLP [to load it](docs/datasets.md#danish-dependency-treebank)
- :performing_arts: [Notebook](examples/example_zero_shot_sentiment.ipynb) tutorial for ​Sentiment analysis models using 
  zero-shot transfer learning from [LASER](https://github.com/facebookresearch/LASER/tree/master/source)
- 🐋 Docker images with all NLP libraries used by DaNLP is available on [DockerHub](https://hub.docker.com/r/alexandrainst/danlp)  

**Next up**

- 🚧 Models trained on the new annotated [Danish NER dataset](docs/datasets.md#danish-dependency-treebank)
- 🚧 Support for Danish in the [spaCy](https://github.com/explosion/spaCy) framework

## Get started
To get started using DaNLP in your python project simply install the pip package. However installing the pip package 
will not install all NLP libraries. If you to try out the models in DaNLP you can use the Docker images
that has all the NLP libraries installed.

### Install with pip
To get started using DaNLP simply install the project with pip:

```bash
pip install danlp
```

Note that the installation of DaNLP does not install other NLP libraries such as Gensim, Spacy or Flair.
This is allows the installation to be as minimal as possible and let the user choose to e.g. load word embeddings
with either spaCy or Gensim.

### Install with Docker 
To quickly get started with DaNLP to try out the models you can use our Docker image.
To start a ipython session simply run:
```bash
docker run -it --rm alexandrainst/danlp ipython
```
If you want to run a `<script.py>` in you current working directory you can run:
```bash
docker run -it --rm -v "$PWD":/usr/src/app -w /usr/src/app alexandrainst/danlp python <script.py>
```
You can also quickly get started with one of our [notebooks](/examples).
  ​                   


## NLP Models
Natural Language Processing is an active area of research and it consists of many different tasks. 
The DaNLP repository provides an overview of Danish models for some of the most common NLP tasks.

The repository is under development and this is the list of NLP tasks we have covered and plan to cover in the repository.
- [x] [Embedding of text](docs/models/embeddings.md)
- [x] [Part of speech](docs/models/pos.md)
- [x] [Named Entity Recognition](docs/models/ner.md)
- [x] [Sentiment Analysis](docs/sentiment_analysis.md)
- [ ] Coreference resolution

If you are interested in Danish support for any specific NLP task you are welcome to get in contact with us.

## Datasets
The number of datasets in the Danish is limited. The DaNLP repository provides and overview of the available Danish datasets that can be used for commercial purposes.

The DaNLP package allows you to download and preprocess datasets. You can read about the datasets [here](/docs/datasets.md).

## Examples
You will find examples and tutorials [here](/examples) that shows how to use NLP in Danish.
We will also provide some real cases of how NLP is applied in Danish companies.

## How do I contribute?

If you want to contribute to the DaNLP repository and make it better, your help is very welcome. You can contribute to the project in many ways:

- Help us write good tutorials on Danish NLP use-cases
- Contribute with you own pretrained NLP models or datasets in Danish
- Notify us of other Danish NLP resources
- Create GitHub issues with questions and bug reports

## Who is behind?
<img align="right" width="150" src="https://raw.githubusercontent.com/alexandrainst/danlp/master/docs/imgs/alexandra_logo.png">

The DaNLP repository is maintained by the [Alexandra Institute](https://alexandra.dk/uk) which is a Danish non-profit company 
with a mission to create value, growth and welfare in society. The Alexandra Institute is a member of [GTS](https://gts-net.dk/), 
a network of independent Danish research and technology organisations.

The work on this repository is part the [Dansk For Alle](https://bedreinnovation.dk/dansk-alle-0) performance contract 
allocated to the Alexandra Insitute by the [Danish Ministry of Higher Education and Science](https://ufm.dk/en?set_language=en&cl=en). The project runs in two years in 2019 and 2020.
