Installation
============


To get started using DaNLP in your python project simply install the pip package. However installing the pip package 
will not install all NLP libraries because we want you to have the freedom to limit the dependency on what you use.

### Install with pip

To get started using DaNLP simply install the project with pip:

```bash
pip install danlp 
```

Note that the installation of DaNLP does not install other NLP libraries such as Gensim, SpaCy, flair or Transformers.
This allows the installation to be as minimal as possible and let the user choose to e.g. load word embeddings with either spaCy, flair or Gensim.  Therefore, depending on the function you need to use, you should install one or several of the following: `pip install flair`, `pip install spacy ` or/and `pip install gensim `. You can check the `requirements.txt` file to see what version the packages has been tested with.

### Install from source

If you want to be able to use the latest developments before they are released in a new pip package, or you want to modify the code yourself, then clone this repo and install from source. 

```
git clone https://github.com/alexandrainst/danlp.git
cd danlp
pip install . 
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