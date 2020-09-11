<h1 align="center">
  <img src="https://raw.githubusercontent.com/alexandrainst/danlp/master/docs/imgs/danlp_logo.png"  width="350"  />
</h1>

<div align="center">
  <a href="https://pypi.org/project/danlp/"><img src="https://img.shields.io/pypi/v/danlp.svg"></a>
  <a href="https://travis-ci.org/alexandrainst/danlp"><img src="https://travis-ci.org/alexandrainst/danlp.svg?branch=master"></a>
  <a href="https://coveralls.io/github/alexandrainst/danlp?branch=master"><img src="https://coveralls.io/repos/github/alexandrainst/danlp/badge.svg?branch=master"></a>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/license-BSD%203-blue.svg"></a>
</div>
<div align="center">
  <h5>
    <a href="https://github.com/alexandrainst/danlp/blob/master/docs/models/ner.md">
      Named Entity Recognition
      </a>
      <span> | </span>
    <a href="https://github.com/alexandrainst/danlp/blob/master/docs/models/pos.md">
      Part of Speech
    </a>
    <span> | </span>
    <a href="https://github.com/alexandrainst/danlp/blob/master/docs/models/sentiment_analysis.md">
      Sentiment Analysis
    </a>
      <span> | </span>
    <a href="https://github.com/alexandrainst/danlp/blob/master/docs/models/embeddings.md">
      Embeddings
      </a>
  </h5>
    <h5>
   	 <a href="https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md">
      Datasets
   	 </a>
      <span> | </span>
   	 <a href="https://github.com/alexandrainst/danlp/tree/master/examples">
      Examples
   	 </a>
  </h5>
</div>
DaNLP is a repository for Natural Language Processing resources for the Danish Language. 
It is a collection  of available datasets and models for a variety of NLP tasks.
It features code examples on how to use the datasets and models in popular NLP frameworks such as spaCy and Flair as well as Deep Learning frameworks such as PyTorch and TensorFlow. 

<br/>**Help us improve DaNLP**
- :raising_hand: Have you tried the DaNLP package? Then we would love to chat with you about your experiences from a company perspective. It will take approx 20-30 minutes and there's no preparation. English/danish as you prefer. Please leave your details [here](https://forms.office.com/Pages/ResponsePage.aspx?id=zSPaS4dKm0GkfXZzEwsohKhC_ON5BmxBtRwkonVf21tUQUxDQ0oyTVAyU0tDUDVDMTM4SkU4SjJISi4u) and then we will reach out to arrange a call. We also welcome and appreciate any written feedback. Reach us at [danlp@alexandra.dk](mailto:danlp@alexandra.dk)

**News**
- :paw_prints: A first version of a Spacy models for sentiment trained using hard distill from BERT is added to the repo, read about it in the [docs](https://github.com/alexandrainst/danlp/blob/master/docs/models/sentiment_analysis.md)
- :hotel: :broken_heart:  Version 0.0.9 has been [released](https://github.com/alexandrainst/danlp/releases) with an update of storage host for models and dataset hosted by danlp - this means older pip version support for downloading models and dataset from danlp host is broken. 
- 🚧 Support for Danish in the [spaCy]( https://spacy.io/models/da) new 2.3 version. The progress for supporting spaCy can be seen here [issue #3056](https://github.com/explosion/spaCy/issues/3056). The spacy model is trained using DaNE and DDT [datasets](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane) - Read more about using spacy through danlp [here](https://github.com/alexandrainst/danlp/blob/master/docs/spacy.md)  

  


**Next up**

- :traffic_light: An attempt to access fairness in sentiment models  through a synthetic test will be added as an example in a Jupyter notebook  

- :paw_prints: Improving spaCy ner model using hard distil of Bert Ner



## Get started

To get started using DaNLP in your python project simply install the pip package. However installing the pip package 
will not install all NLP libraries. If you want to try out the models in DaNLP you can use the Docker images
that has all the NLP libraries installed.

### Install with pip
To get started using DaNLP simply install the project with pip:

```bash
pip install danlp 
```

Note that the installation of DaNLP does not install other NLP libraries such as Gensim, Spacy or Flair.
This allows the installation to be as minimal as possible and let the user choose to e.g. load word embeddings with either spaCy, flair or Gensim.  Therefore, depending on the function you need to use, you should install one or several of the following: `pip install flair`, `pip install spacy ` or/and `pip install gensim `.

### Install from source

If you want to be able to use the latest developments before they are realized in a new pip package, or you want to modify the code your self, then clone this repo and install from source. 

```
git clone https://github.com/alexandrainst/danlp.git
cd danlp
pip install . 
```

if you have clone it before use ``git pull`` to get newest version instead of clone.  

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
You can also quickly get started with one of our [notebooks](/examples).
  ​                   


## NLP Models
Natural Language Processing is an active area of research and it consists of many different tasks. 
The DaNLP repository provides an overview of Danish models for some of the most common NLP tasks.

The repository is under development and this is the list of NLP tasks we have covered and plan to cover in the repository.
-  [Embedding of text](docs/models/embeddings.md)
-  [Part of speech](docs/models/pos.md)
-  [Named Entity Recognition](docs/models/ner.md)
-  [Sentiment Analysis](docs/models/sentiment_analysis.md)
-  Coreference resolution

If you are interested in Danish support for any specific NLP task you are welcome to get in contact with us.

We do also recommend to check out this awesome [list](https://github.com/fnielsen/awesome-danish) of Danish NLP stuff from Finn Årup Nielsen. 

## Datasets
The number of datasets in the Danish is limited. The DaNLP repository provides an overview of the available Danish datasets that can be used for commercial purposes.

The DaNLP package allows you to download and preprocess datasets. You can read about the datasets [here](/docs/datasets.md).

## Examples
You will find examples and tutorials [here](/examples) that shows how to use NLP in Danish. This project keeps a Danish written [blog](https://medium.com/danlp) on medium where we write about Danish NLP, and in time we will also provide some real cases of how NLP is applied in Danish companies.

## How do I contribute?

If you want to contribute to the DaNLP repository and make it better, your help is very welcome. You can contribute to the project in many ways:

- Help us write good tutorials on Danish NLP use-cases
- Contribute with your own pretrained NLP models or datasets in Danish
- Notify us of other Danish NLP resources
- Create GitHub issues with questions and bug reports

## Who is behind?
<img align="right" width="150" src="https://raw.githubusercontent.com/alexandrainst/danlp/master/docs/imgs/alexandra_logo.png">

The DaNLP repository is maintained by the [Alexandra Institute](https://alexandra.dk/uk) which is a Danish non-profit company 
with a mission to create value, growth and welfare in society. The Alexandra Institute is a member of [GTS](https://gts-net.dk/), 
a network of independent Danish research and technology organisations.

The work on this repository is part the [Dansk For Alle](https://bedreinnovation.dk/dansk-alle-0) performance contract 
allocated to the Alexandra Insitute by the [Danish Ministry of Higher Education and Science](https://ufm.dk/en?set_language=en&cl=en). The project runs in two years in 2019 and 2020, and an overview  of the project can be found on our [microsite](https://danlp.alexandra.dk/). ````
