Datasets
========

This section keeps a list of Danish NLP datasets publicly available. 

| Dataset                                                      | Task                   | Words             | Sents                  | License                                                      | DaNLP |
| ------------------------------------------------------------ | ---------------------- | ----------------- | ---------------------- | ------------------------------------------------------------ | ----- |
| [OpenSubtitles2018](<http://opus.nlpl.eu/OpenSubtitles2018.php>) | Translation            | 206,700,000       | 30,178,452             | [None](http://opus.nlpl.eu/OpenSubtitles2018.php)            | ❌     |
| [EU Bookshop](http://opus.nlpl.eu/EUbookshop-v2.php)         | Translation            | 208,175,843       | 8,650,537              | -                                                            | ❌     |
| [Europarl7](http://www.statmt.org/europarl/)                 | Translation            | 47,761,381        | 2,323,099              | [None](http://www.statmt.org/europarl/)                      | ❌     |
| [ParaCrawl5](https://paracrawl.eu/)                          | Translation            | -                 | -                      | [CC0](https://paracrawl.eu/releases.html)                    | ❌     |
| [WikiANN](#wikiann)                                          | NER                    | 832.901           | 95.924                 | [ODC-BY 1.0](http://nlp.cs.rpi.edu/wikiann/)                 | ✔️     |
| [UD-DDT (DaNE)](#dane)                                       | DEP, POS, NER          | 100,733           | 5,512                  | [CC BY-SA 4.0](https://github.com/UniversalDependencies/UD_Danish-DDT/blob/master/README.md) | ✔️     |
| [LCC Sentiment](#lcc-sentiment)                              | Sentiment              | 10.588            | 499                    | [CC BY](https://github.com/fnielsen/lcc-sentiment/blob/master/LICENSE) | ✔️     |
| [Europarl Sentiment1](#europarl-sentiment1)                  | Sentiment              | 3.359             | 184                    | None                                                         | ✔️     |
| [Europarl Sentiment2](#europarl-sentiment2)                  | sentiment              |                   | 957                    | CC BY-SA 4.0                                                 | ✔️     |
| [Wikipedia](https://dumps.wikimedia.org/dawiki/latest/)      | Raw                    | -                 | -                      | [CC BY-SA 3.0](https://dumps.wikimedia.org/legal.html)       | ❌     |
| [WordSim-353](#wordsim-353)                                  | Word Similarity        | 353               | -                      | [CC BY 4.0](https://github.com/fnielsen/dasem/blob/master/dasem/data/wordsim353-da/LICENSE) | ✔️     |
| [Danish Similarity Dataset](#danish-similarity-dataset)      | Word Similarity        | 99                | -                      | [CC BY 4.0](https://github.com/fnielsen/dasem/blob/master/dasem/data/wordsim353-da/LICENSE) | ✔️     |
| [Twitter Sentiment](#twitter-sentiment)                      | Sentiment              | -                 | train: 1215, test: 512 | Twitter privacy policy applies                               | ✔️     |
| [Dacoref](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#dacoref) | coreference resolution | 64.076 (tokens)   | 3.403                  | CC BY-SA 4.0                                 | ✔️     |
| [DanNet](#dannet)                                            | Wordnet                | 66.308 (concepts) | -                      | [license](https://cst.ku.dk/projekter/dannet/license.txt)    | ✔️     |

It is also recommend to check out Finn Årup Nielsen's [dasem github](https://github.com/fnielsen/dasem) which also provides script for loading different Danish corpus. 

### Danish Dependency Treebank (DaNE) {#dane}

The Danish UD treebank (Johannsen et al., 2015, UD-DDT) is a
conversion of the Danish Dependency Treebank (Buch-Kromann et
al. 2003) based on texts from Parole (Britt, 1998).
UD-DDT has annotations for dependency parsing and part-of-speech (POS) tagging. 
The dataset was annotated with Named Entities for **PER**, **ORG** and **LOC** 
by the Alexandra Institute in the DaNE dataset (Hvingelby et al. 2020).
To read more about how the dataset was annotated with POS and DEP tags we refer to the
[Universal Dependencies](https://github.com/UniversalDependencies/UD_Danish-DDT/) page.
The dataset can be loaded with the DaNLP package:

```python
from danlp.datasets import DDT
ddt = DDT()

spacy_corpus = ddt.load_with_spacy()
flair_corpus = ddt.load_with_flair()
conllu_format = ddt.load_as_conllu()
```

The dataset can also be downloaded directly in CoNLL-U format.

[Download DDT](https://danlp.alexandra.dk/304bd159d5de/datasets/ddt.zip) 

### Dacoref

This Danish coreference annotation contains parts of the Copenhagen Dependency Treebank  (Kromann and Lynge, 2004), It was originally annotated as part of the Copenhagen Dependency Treebank (CDT) project but never finished. This resource extends the annotation by using different mapping techniques and by augmenting with Qcodes from Wiktionary. This work is conducted by Maria Jung Barrett. Read more about it in the dedicated [dacoref docs](dacoref_docs.md).

The dataset can be loaded with the DaNLP package:

```python
from danlp.datasets import Dacoref
dacoref = Dacoref()
# The corpus can be loaded with or without splitting into train, dev and test in a list in that order
corpus = dacoref.load_as_conllu(predefined_splits=True) 
```

The dataset can also be downloaded directly:

[Download dacoref](http://danlp-downloads.alexandra.dk/datasets/dacoref.zip) 

### WikiANN
The WikiANN dataset [(Pan et al. 2017)](https://aclweb.org/anthology/P17-1178) is a dataset with NER annotations 
for **PER**, **ORG** and **LOC**. It has been constructed using the linked entities in Wikipedia pages for 282 different
languages including Danish. The dataset can be loaded with the DaNLP package: 

```python
from danlp.datasets import WikiAnn
wikiann = WikiAnn()

spacy_corpus = wikiann.load_with_spacy()
flair_corpus = wikiann.load_with_flair()
```

### WordSim-353
The WordSim-353 dataset (Finkelstein et al. 2002) contains word pairs annotated with a similarity score (1-10). It is common to use it to do intrinsic evaluations 
on word embeddings to test for syntactic or semantic relationships between words. The dataset has been 
[translated to Danish](https://github.com/fnielsen/dasem/tree/master/dasem/data/wordsim353-da) by Finn Årup Nielsen.

### Danish Similarity Dataset
The [Danish Similarity Dataset](https://github.com/kuhumcst/Danish-Similarity-Dataset) 
consists of 99 word pairs annotated by 38 annotators with a similarity score (1-6).
It is constructed with frequently used Danish words.

### Twitter Sentiment

The Twitter sentiment is original a small manually annotated dataset by the Alexandra Institute conducted by one trained annotator. It contains tags in two sentiment dimension: analytic: ['subjective' , 'objective'] and polarity: ['positive', 'neutral', 'negative' ]. It is split in train and test part. The train part is denoted as training_version 1.

Annotations for polarity is extend through a crowd-sourcing game named Angry Tweets. Here, volunteers were asked to annotate tweets based on what they thought the authors had "meant, felt or thought". An option for "skip" provided in case of the tweet being, e.g. not Danish. In an attempt to control the annotation quality and increase the gamification element, Game Overs and points were issued in the following way. The game (one session) consisted of eight rounds of four tweets per pages. Time was measured on each page, and Game Over was issued if the player was too quick. On one out of every two pages, one of the tweets was a "verifying" tweet which had prior to the game been annotated with agreement by four trained annotators. Game Over was issued if the players' annotation did not match. On every page, one tweet was previous annotated, and if the players' annotation match, one point was granted. In addition, one point was granted for completing a round. The annotation for the training_version 1 was used in the game to initial the double annotation, and therefore part of training_version 2. Downloading the training version 2 provides information of the annotations, the sessions with point and time, the game overs and the verifying tweets. 

Due to Twitter's privacy policy, it is only allowed to display the "tweet ID" and not the actual text. This will enable people to delete their tweets. Therefore, to download the actual tweet text, one needs a Twitter development account, and to generate the sets of login keys, read how to get started [here](https://python-twitter.readthedocs.io/en/latest/getting_started.html). Then the dataset can be loaded with the DaNLP package by setting the following environment variable for the keys:

``` TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET```

```python
from danlp.datasets import TwitterSent
twitSent = TwitterSent()
# the default is to download the training version 1 which is annotated by a trained annotator
df_test, df_train = twitSent.load_with_pandas()
# To download the tweets annotated in the game and the ekstra information
df_test, df_train, df_gameover, df_verifying, df_session = 	twitSent.load_with_pandas( 	 training_version=2, game_info=True)
```

The dataset can also be downloaded directly with the labels and tweet id:

​	Test set and first training version: [Download TwitterSent](https://danlp-downloads.alexandra.dk/datasets/twitter.sentiment.zip) 

​	Second training version with extra annotation from the game: [Download TwitterSent - game](https://danlp-downloads.alexandra.dk/datasets/game_tweets.zip) 

​	Information about the game annotations: [Download TwitterSent - game information](https://danlp-downloads.alexandra.dk/datasets/AngryTweets.zip) 

### Europarl Sentiment1

The [Europarl Sentiment1](https://github.com/fnielsen/europarl-da-sentiment) dataset contains sentences from 
the [Europarl](http://www.statmt.org/europarl/) corpus which has been annotated manually by Finn Årup Nielsen.
Each sentence has been annotated the polarity of the sentiment as an polarity score from -5 to 5. 
The score can be converted to positive (>0), neutral (=0) and negative (<0). 
The dataset can be loaded with the DaNLP package:

```python
from danlp.datasets import EuroparlSentiment1
eurosent = EuroparlSentiment1()

df = eurosent.load_with_pandas()
```

### Europarl Sentiment2

The dataset consist of  957 manually annotation by Alexandra institute on sentences from Eruroparl. It contains tags in two sentiment dimension: analytic: ['subjective' , 'objective'] and polarity: ['positive', 'neutral', 'negative' ]. 
The dataset can be loaded with the DaNLP package:

```python
from danlp.datasets import EuroparlSentiment2
eurosent = EuroparlSentiment2()

df = eurosent.load_with_pandas()
```

### LCC Sentiment

The [LCC Sentiment](https://github.com/fnielsen/lcc-sentiment) dataset contains sentences from Leipzig Copora Collection [(Quasthoff et al. 2006)](https://www.aclweb.org/anthology/L06-1396/) 
which has been manually annotated by Finn Årup Nielsen.  
Each sentence has been annotated the polarity of the sentiment as an polarity score from -5 to 5.
The score can be converted to positive (>0), neutral (=0) and negative (<0).
The dataset can be loaded with the DaNLP package:
```python
from danlp.datasets import LccSentiment
lccsent = LccSentiment()

df = lccsent.load_with_pandas()
```


### DanNet

[DanNet](https://cst.ku.dk/projekter/dannet/) is a lexical database such as [Wordnet](https://wordnet.princeton.edu/). "Center for sprogteknologi" at The University of Copenhagen is behind it and more details about it can be found in the paper (Pedersen et al 2009).

DanNet depicts the relations between words in Danish (mostly nouns, verbs and adjectives). 
The main relation among words in WordNet is synonymy.

The dataset consists of 4 databases:

    * words
    * word senses
    * relations
    * synsets

DanNet uses the concept of `synset` to link words together. All the words in the database are part of one or multiple synsets. A synset is a set of synonyms (words which have the same meanings).


For downloading DanNet through DaNLP, you can do: 

```python
from danlp.datasets import DanNet

dannet = DanNet()

# you can load the databases if you want to look into the databases by yourself
words, wordsenses, relations, synsets = dannet.load_with_pandas()
```

We also provide helper functions to search for synonyms, hyperonyms, hyponyms and domains through the databases. 
Once you have downloaded the DanNet wrapper, you can use the following features: 

```python

word = "myre"
# synonyms
dannet.synonyms(word)
""" ['tissemyre'] """
# hypernyms
dannet.hypernyms(word)
""" ['årevingede insekter'] """
# hyponyms
dannet.hyponyms(word)
""" ['hærmyre', 'skovmyre', 'pissemyre', 'tissemyre'] """
# domains
dannet.domains(word)
""" ['zoologi'] """
# meanings
dannet.meanings(word)
""" ['ca. 1 cm langt, årevinget insekt med en kraftig in ... (Brug: "Myrer på terrassen, og andre steder udendørs, kan hurtigt blive meget generende")'] """


# to help you dive into the databases
# we also provide the following functions: 

# part-of-speech (returns a list comprised in 'Noun', 'Verb' or 'Adjective')
dannet.pos(word)
# wordnet relations (EUROWORDNET or WORDNETOWL)
dannet.wordnet_relations(word, eurowordnet=True))
# word ids
dannet._word_ids(word)
# synset ids
dannet._synset_ids(word)
# word from id
dannet._word_from_id(11034863)
# synset from id
dannet._synset_from_id(3514)
```


## 🎓 References
- Johannsen, Anders, Martínez Alonso, Héctor and Plank, Barbara. [Universal Dependencies for Danish](http://tlt14.ipipan.waw.pl/files/4914/4974/3227/TLT14_proceedings.pdf#page=164). TLT14, 2015.
- Keson, Britt (1998). [Documentation of The Danish Morpho-syntactically Tagged PAROLE Corpus](https://korpus.dsl.dk/clarin/corpus-doc/parole-doc/paroledoc_en.pdf). Technical report, DSL
- Matthias T. Buch-Kromann, Line Mikkelsen, and Stine Kern Lynge. 2003. [Danish dependency treebank](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.208.6716&rep=rep1&type=pdf). In **TLT**.
- Rasmus Hvingelby, Amalie B. Pauli, Maria Barrett, Christina Rosted, Lasse M. Lidegaard and Anders Søgaard. 2020. [DaNE: A Named Entity Resource for Danish](https://www.aclweb.org/anthology/2020.lrec-1.565.pdf). In **LREC**.
- Pedersen, Bolette S. Sanni Nimb, Jørg Asmussen, Nicolai H. Sørensen, Lars Trap-Jensen og Henrik Lorentzen (2009). [DanNet – the challenge of compiling a WordNet for Danish by reusing a monolingual dictionary](https://pdfs.semanticscholar.org/6891/69de00c63d58bd68229cb0b3469a617f5ab3.pdf). *Lang Resources & Evaluation* 43:269–299.
- Xiaoman Pan, Boliang Zhang, Jonathan May, Joel Nothman, Kevin Knight and Heng Ji. 2017. [Cross-lingual Name Tagging and Linking for 282 Languages](https://aclweb.org/anthology/P17-1178). In **ACL**.
- Lev Finkelstein, Evgeniy Gabrilovich, Yossi Matias, Ehud Rivlin, Zach Solan, Gadi Wolfman, and Eytan Ruppin. 2002. [Placing Search in Context: The Concept Revisited](http://www.cs.tau.ac.il/~ruppin/p116-finkelstein.pdf). In  **ACM TOIS**.
- Uwe Quasthoff, Matthias Richter and Christian Biemann. 2006. [Corpus Portal for Search in Monolingual Corpora](https://www.aclweb.org/anthology/L06-1396/). In **LREC**.
-  M.T. Kromann and S.K. Lynge. [Danish Dependency Treebank v. 1.0](https://github.com/mbkromann/copenhagen-dependency-treebank). Department of Computational Linguistics, Copenhagen Business School., 2004. 
