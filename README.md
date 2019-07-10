# <img src="docs/imgs/danlp_logo.png"  width="450"  /> 

[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Build Status](https://travis-ci.org/alexandrainst/danlp.svg?branch=master)](https://travis-ci.org/alexandrainst/danlp)

DaNLP is a repository for Natural Language Processing resources for the Danish Language. It is a collection  of available datasets and models for a a variety of NLP tasks. It features code examples on how to use the datasets andmodels in popular NLP frameworks such as spaCy and NLTK as well as Deep Learning frameworks such as PyTorch and TensorFlow.

The aim for this project is both to point to the open source tool available in Danish, and to add new models and tools to make NLP in Danish more applicable for everybody and especially for the industry.



#### News

- DaNLP can now be installed as a pip package. 

#### Next up

This project is currently working on sentiment analysis using zero-shot transfer learning from [LASER](https://github.com/facebookresearch/LASER/tree/master/source), looking into weak supervision, a tutorial for using transfer learning with [BERT](https://github.com/google-research/bert),  and soon it will begin to look at how to support SpaCy for Danish. 

## Get started
##### Instal with pip

The project is build for Python 3.6+. To get started you need to clone the project and install it with pip:

```bash
git clone git@github.com:alexandrainst/danlp.git
cd danlp
pip install .
```

The DaNLP package wraps existing NLP models for Danish, and provides scripts for downloading Danish datasets.

Note that the installation of DaNLP does not install other packages used in different task such as Gensim, Spacy or Flair. This is to allow the installation to be as clean as possible and give the user the freedom to chose if for example the word embedding should be loaded with Spacy or Gensim. 

##### Install with Docker 

This installation option uses Docker and builds an image with all the used packages in this repository. Choose this option if you want to get started fast without the need to manually install extra packages. 

- Start by getting and install Docker CE for your system (Linux, OSX or Windows):
  https://docs.docker.com/install/
  Maybe run a "hello world" to ensure it is working properly.

- Clone the DaNLP repository and navigate to the folder:

    ```bash
    git clone git@github.com:alexandrainst/danlp.git
    cd danlp
    ```

- Create the Docker image (which you only need to do once), run: 

  ```bash
  docker build -t image_danlp . 
  ```


- Create a container for the first time:

  - On Linux and OSX, run: 	

    ```bash
    docker run -it  --name container_danlp -p 8888:8888 -v $PWD:/root image_danlp
    ```

  - On Windows, change username and path to project folder and run:

    ```bash
    docker run -it --name container_danlp -p 8888:8888 --user root -v /c/Users/$YOUR_USERNAME/path/to/project/folder:/root/ image_danlp
    ```

  - On WSL, change username and path to project folder and run:

    ```bash
    docker run -it --name container_danlp -p 8888:8888 --user root -v "C:\Users\YOUR_USERNAME\path\to\project\folder":/root/  image_danlp
    ```

- Inside the container, install the DaNLP package:

  ```bash
  install danlp . 
  ```

- Now you can are ready to run python3 scripts and commands or use Jupyter notebook to run the code examples.

   - To use Jupyter Notebook, run:

     ```docker
     jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
     ```

  - Open the notebook in a browser using `http://localhost:8888/`  and pass the token.

- You can stop and resume the container by  `docker stop container_danlp` and `docker start -i container_danlp`



## Docs 

The documentation aims to provide the following:

- Overview for each task regarding what is available in Danish in open source preferable with benchmark results

- Introduction and examples of how to run the code and use NLP for the Danish Languages and in time real case stories 

- Explanation and training details for models trained and datasets created in this project

  ​                   


## NLP Models
Natural Language Processing is an active area of research and it consists of many different tasks. The DaNLP repository provides an overview of Danish models for some of the most common NLP tasks.

The repository is under development and this is the list of NLP tasks planned to be covered in the repository.
- [x] [Embedding of text](docs/models/embeddings.md)
- [x] [Part of speech](docs/models/part_of_speach_tagging.md)
- [ ] Named Entity Recognition
- [ ] Lemmatization
- [ ] Sentiment Analysis
- [ ] Coreference resolution

If you are interessted in supporting Danish for any specific NLP task you are welcome to get in contact with us.

## Datasets
The number of datasets in the Danish is limited. The DaNLP repository provides and overview of the available Danish datasets that can be used for commercial purposes.

The DaNLP package allows you to download and preprocess datasets. You can read about the datasets [here](/docs/datasets.md).

## Examples
You will find examples and small tutorials [here](/examples/examples.md) that shows how to use NLP in Danish.
We will also provide some real cases of how NLP is applied in Danish companies.

## How do I contribute?

If you want to contribute to the DaNLP repository and make it better, your help is very welcome. You can contribute to the project in many ways:

- Help us write good tutorials on Danish NLP use-cases
- Contribute with you own pretrained NLP models or datasets in Danish
- Notify us of other Danish NLP resources
- Create GitHub issues with questions and bug reports

## Who is behind?
<img align="right" width="150" src="docs/imgs/alexandra_logo.png">

The DaNLP repository is maintained by the [Alexandra Institute](https://alexandra.dk/uk) which is a Danish non-profit company 
with a mission to create value, growth and welfare in society. The Alexandra Institute is a member of [GTS](https://gts-net.dk/), 
a network of independent Danish research and technology organisations.

The work on this repository is part the [Dansk For Alle](https://bedreinnovation.dk/dansk-alle-0) performance contract 
allocated to the Alexandra Insitute by the [Danish Ministry of Higher Education and Science](https://ufm.dk/en?set_language=en&cl=en). The project runs in two years in 2019 and 2020.
