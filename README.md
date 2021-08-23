<h1 align="center">
  <img src="https://raw.githubusercontent.com/alexandrainst/danlp/master/docs/docs/imgs/danlp_logo.png"  width="350"  />
</h1>

<div align="center">
  <a href="https://pypi.org/project/danlp/"><img src="https://img.shields.io/pypi/v/danlp.svg"></a>
  <a href="https://travis-ci.com/alexandrainst/danlp"><img src="https://travis-ci.com/alexandrainst/danlp.svg?branch=master"></a>
  <a href="https://coveralls.io/github/alexandrainst/danlp?branch=master"><img src="https://coveralls.io/repos/github/alexandrainst/danlp/badge.svg?branch=master"></a>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/license-BSD%203-blue.svg"></a>
  <a href='https://danlp-alexandra.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/danlp-alexandra/badge/?version=latest' alt='Documentation Status' /></a>
</div>
<div align="center">
  <h5>
    <a href="https://github.com/alexandrainst/danlp/blob/master/docs/docs/tasks/ner.md">
      Named Entity Recognition
      </a>
      <span> | </span>
    <a href="https://github.com/alexandrainst/danlp/blob/master/docs/docs/tasks/pos.md">
      Part of Speech Tagging
    </a>
    <span> | </span>
    <a href="https://github.com/alexandrainst/danlp/blob/master/docs/docs/tasks/dependency.md">
      Dependency Parsing
    </a>
    <span> | </span>
    <a href="https://github.com/alexandrainst/danlp/blob/master/docs/docs/tasks/sentiment_analysis.md">
      Sentiment Analysis
    </a>
      <span> | </span>
    <a href="https://github.com/alexandrainst/danlp/blob/master/docs/docs/tasks/coreference.md">
      Coreference Resolution
      </a>
      <span> | </span>
    <a href="https://github.com/alexandrainst/danlp/blob/master/docs/docs/tasks/hatespeech.md">
      Hatespeech Detection
      </a>
      <span> | </span>
    <a href="https://github.com/alexandrainst/danlp/blob/master/docs/docs/tasks/embeddings.md">
      Embeddings
      </a>
  </h5>
    <h5>
   	 <a href="https://github.com/alexandrainst/danlp/blob/master/docs/docs/datasets.md">
      Datasets
   	 </a>
      <span> | </span>
   	 <a href="https://github.com/alexandrainst/danlp/tree/master/examples/tutorials">
      Tutorials
   	 </a>
  </h5>
</div>

DaNLP is a repository for Natural Language Processing resources for the Danish Language. 
It is a collection  of available datasets and models for a variety of NLP tasks. 
The aim is to make it easier and more applicable to practitioners in the industry to use 
Danish NLP and hence this project is licensed to allow commercial use. 
The project features code examples on how to use the datasets and models in popular 
NLP frameworks such as spaCy, Transformers and Flair as well as Deep Learning frameworks 
such as PyTorch. 
See our [documentation pages](https://danlp-alexandra.readthedocs.io/en/latest/index.html) 
for more details about our models and datasets, and definitions of the modules provided 
through the DaNLP package. 

If you are new to NLP or want to know more about the project in a broader perspective, you can start on our [microsite](https://danlp.alexandra.dk/).


<br/>**Help us improve DaNLP**

- :raising_hand: Have you tried the DaNLP package? Then we would love to chat with you about your experiences from a company perspective. It will take approx 20-30 minutes and there's no preparation. English/danish as you prefer. Please leave your details [here](https://forms.office.com/Pages/ResponsePage.aspx?id=zSPaS4dKm0GkfXZzEwsohKhC_ON5BmxBtRwkonVf21tUQUxDQ0oyTVAyU0tDUDVDMTM4SkU4SjJISi4u) and then we will reach out to arrange a call. 

**News**

- :tada: Version 0.0.12 has been [released](https://github.com/alexandrainst/danlp/releases) with a new feature for finding the `domains` of a word with DanNet. One new dataset has been made available -- the DKHate -- for hate speech detection and one new model has been added for coreference resolution.  

- :blue_book: A jupyter notebook tutorial for quick starting
- Benchmarking of the NERDA model for NER
- Benchmarking of the Dacy model for PoS-tagging, NER and dependency parsing


**Next up**

- the AngryTweets dataset will be added to the DaNLP package, as a result of the crowdsourcing game [AngryTweets](https://medium.com/danlp/angry-tweets-f%C3%B8lelser-og-annoteringer-er-p%C3%A5-spil-s%C3%A5-spil-med-eacade042c95)


## Installation

To get started using DaNLP in your python project simply install the pip package. Note that installing the default pip package 
will not install all NLP libraries because we want you to have the freedom to limit the dependency on what you use. Instead we provide you with an installation option if you want to install all the required dependencies. 

### Install with pip

To get started using DaNLP simply install the project with pip:

```bash
pip install danlp 
```

Note that the default installation of DaNLP does not install other NLP libraries such as Gensim, SpaCy, flair or Transformers.
This allows the installation to be as minimal as possible and let the user choose to e.g. load word embeddings with either spaCy, flair or Gensim.  Therefore, depending on the function you need to use, you should install one or several of the following: `pip install flair`, `pip install spacy ` or/and `pip install gensim `. 

Alternatively if you want to install all the required dependencies including the packages mentionned above, you can do:

```bash
pip install danlp[all]
```

You can check the `requirements.txt` file to see what version the packages has been tested with.

### Install from source

If you want to be able to use the latest developments before they are released in a new pip package, or you want to modify the code yourself, then clone this repo and install from source.

```
git clone https://github.com/alexandrainst/danlp.git
cd danlp
# minimum installation
pip install .
# or install all the packages
pip install .[all]
```

To install the dependencies used in the package with the tested versions:

```python
pip install -r requirements.txt
```


### Install from github
Alternatively you can install the latest version from github using:
```
pip install git+https://github.com/alexandrainst/danlp.git
```

### Install with Docker 
To quickly get started with DaNLP and to try out the models you can use our Docker image.
To start a ipython session simply run:
```bash
docker run -it --rm alexandrainst/danlp ipython
```
If you want to run a `<script.py>` in your current working directory you can run:
```bash
docker run -it --rm -v "$PWD":/usr/src/app -w /usr/src/app alexandrainst/danlp python <script.py>
                  
```

### Quick Start

Read more in our [documentation pages](https://danlp-alexandra.readthedocs.io/en/latest/docs/gettingstarted/quickstart.html).


## NLP Models

Natural Language Processing is an active area of research and it consists of many different tasks. 
The DaNLP repository provides an overview of Danish models for some of the most common NLP tasks.

The repository is continuously evolving and here is the list of [NLP tasks](https://danlp-alexandra.readthedocs.io/en/latest/tasks.html) we currently cover in the repository.
-  [Embedding of text](docs/docs/tasks/embeddings.md)
-  [Part of speech](docs/docs/tasks/pos.md)
-  [Named entity recognition](docs/docs/tasks/ner.md)
-  [Sentiment analysis](docs/docs/tasks/sentiment_analysis.md)
-  [Dependency parsing](docs/docs/tasks/dependency.md)
-  [Coreference resolution](docs/docs/tasks/coreference.md)
- [Hatespeech detection](docs/docs/tasks/hatespeech.md)

You can also find some of our [transformers](docs/docs/frameworks/transformers.md) models on [HuggingFace](https://huggingface.co/DaNLP). 

If you are interested in Danish support for any specific NLP task you are welcome to get in contact with us.

We do also recommend to check out this awesome [list](https://github.com/fnielsen/awesome-danish) of Danish NLP stuff from Finn Årup Nielsen. 

## Datasets
The number of datasets in the Danish language is limited. The DaNLP repository provides an overview of the available Danish datasets that can be used for commercial purposes.

The DaNLP package allows you to download and preprocess datasets. You can read about the datasets [here](https://danlp-alexandra.readthedocs.io/en/latest/docs/datasets.html).

## Examples
You will find examples that shows how to use NLP in Danish (using our models or others) in our [benchmark](/examples/benchmarks) scripts and jupyter notebook [tutorials](/examples/tutorials). This project keeps a Danish written [blog](https://medium.com/danlp) on medium where we write about Danish NLP, and in time we will also provide some real cases of how NLP is applied in Danish companies. 

## Structure of the repo

To help you navigate we provide you with an overview of the structure in the github:

    .
    ├── danlp		   			# Source files
    │	├── datasets   			# Code to load datasets with different frameworks 
    │	└── models     			# Code to load models with different frameworks 
    ├── docker         			# Docker image
    ├── docs	       			# Documentation and files for setting up Read The Docs
    │   ├── docs	   			# Documentation for tasks, datasets and frameworks
    │	    ├── tasks  			# Documentation for nlp tasks with benchmark results
    │	    ├── frameworks 		# Overview over different frameworks used
    │		├── gettingstarted 	  # Guides for installation and getting started  
    │	    └── imgs   			 # Images used in documentation
    │   └── library     		# Files used for Read the Docs
    ├── examples	   			# Examples, tutorials and benchmark scripts
    │   ├── benchmarks 			# Scripts for reproducing benchmarks results
    │   └── tutorials 			# Jupyter notebook tutorials
    └── tests   	   			# Tests for continuous integration with travis

## How do I contribute?

If you want to contribute to the DaNLP repository and make it better, your help is very welcome. You can contribute to the project in many ways:

- Help us write good [tutorials](examples/tutorials) on Danish NLP use-cases
- Contribute with your own pretrained NLP models or datasets in Danish (see our [contributing guidelines](CONTRIBUTING.md) for more details on how to contribute to this repository)
- Create GitHub issues with questions and bug reports 
- Notify us of other Danish NLP resources or tell us about any good ideas that you have for improving the project through the [Discussions](https://github.com/alexandrainst/danlp/discussions) section of this repository.


## Who is behind?
<img align="right" width="150" src="https://raw.githubusercontent.com/alexandrainst/danlp/master/docs/docs/imgs/alexandra_logo.png">

The DaNLP repository is maintained by the [Alexandra Institute](https://alexandra.dk/uk) which is a Danish non-profit company 
with a mission to create value, growth and welfare in society. The Alexandra Institute is a member of [GTS](https://gts-net.dk/), 
a network of independent Danish research and technology organisations.

Between 2019 and 2020, the work on this repository was part of the [Dansk For Alle](https://bedreinnovation.dk/dansk-alle-0) performance contract (RK) allocated to the Alexandra Institute by the [Danish Ministry of Higher Education and Science](https://ufm.dk/en?set_language=en&cl=en). 
Since 2021, the project is funded through the [Dansk NLP](http://bedreinnovation.dk/dansk-nlp) activity plan which is part of the [Digital sikkerhed, tillid og dataetik](http://bedreinnovation.dk/digital-sikkerhed-tillid-og-dataetik-0) performance contract.

An overview  of the project can be found on our [microsite](https://danlp.alexandra.dk/).

## Cite

If you want to cite this project, please use the following BibTeX entry: 

```
@inproceedings{danlp2021,
    title = {{DaNLP}: An open-source toolkit for Danish Natural Language Processing},
    author = {Brogaard Pauli, Amalie  and
      Barrett, Maria  and
      Lacroix, Ophélie  and
      Hvingelby, Rasmus},
    booktitle = {Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa 2021)},
    month = june,
    year = "2021"
}
```

Read the paper [here](https://ep.liu.se/ecp/178/053/ecp2021178053.pdf). 