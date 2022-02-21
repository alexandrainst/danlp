Datasets
========

This section keeps a list of Danish NLP datasets publicly available. 

| Dataset                                                                                | Task                     | Words             | Sentences              | License                                                                                      | DaNLP |
|----------------------------------------------------------------------------------------|--------------------------|-------------------|------------------------|----------------------------------------------------------------------------------------------|-------|
| [OpenSubtitles2018](<http://opus.nlpl.eu/OpenSubtitles2018.php>)                       | Translation              | 206,700,000       | 30,178,452             | [None](http://opus.nlpl.eu/OpenSubtitles2018.php)                                            | ❌     |
| [EU Bookshop](http://opus.nlpl.eu/EUbookshop-v2.php)                                   | Translation              | 208,175,843       | 8,650,537              | -                                                                                            | ❌     |
| [Europarl7](http://www.statmt.org/europarl/)                                           | Translation              | 47,761,381        | 2,323,099              | [None](http://www.statmt.org/europarl/)                                                      | ❌     |
| [ParaCrawl5](https://paracrawl.eu/)                                                    | Translation              | -                 | -                      | [CC0](https://paracrawl.eu/releases.html)                                                    | ❌     |
| [WikiANN](#wikiann)                                                                    | NER                      | 832,901           | 95,924                 | [ODC-BY 1.0](http://nlp.cs.rpi.edu/wikiann/)                                                 | ✔️    |
| [UD-DDT (DaNE)](#dane)                                                                 | DEP, POS, NER            | 100,733           | 5,512                  | [CC BY-SA 4.0](https://github.com/UniversalDependencies/UD_Danish-DDT/blob/master/README.md) | ✔️    |
| [LCC Sentiment](#lcc-sentiment)                                                        | Sentiment                | 10,588            | 499                    | [CC BY](https://github.com/fnielsen/lcc-sentiment/blob/master/LICENSE)                       | ✔️    |
| [Europarl Sentiment1](#europarl-sentiment1)                                            | Sentiment                | 3,359             | 184                    | None                                                                                         | ✔️    |
| [Europarl Sentiment2](#europarl-sentiment2)                                            | sentiment                |                   | 957                    | CC BY-SA 4.0                                                                                 | ✔️    |
| [Wikipedia](https://dumps.wikimedia.org/dawiki/latest/)                                | Raw                      | -                 | -                      | [CC BY-SA 3.0](https://dumps.wikimedia.org/legal.html)                                       | ❌     |
| [WordSim-353](#wordsim-353)                                                            | Word Similarity          | 353               | -                      | [CC BY 4.0](https://github.com/fnielsen/dasem/blob/master/dasem/data/wordsim353-da/LICENSE)  | ✔️    |
| [Danish Similarity Dataset](#danish-similarity-dataset)                                | Word Similarity          | 99                | -                      | [CC BY 4.0](https://github.com/fnielsen/dasem/blob/master/dasem/data/wordsim353-da/LICENSE)  | ✔️    |
| [Twitter Sentiment](#twitter-sentiment)                                                | Sentiment                | -                 | train: 1,215 -- test: 512 | Twitter privacy policy applies                                                               | ✔️    |
| [AngryTweets](#angrytweets)                                                            | Sentiment                | -                 | 1,266                  | Twitter privacy policy applies                                                               | ✔️    |
| [DaCoref](#dacoref) | coreference resolution   | 64,076 (tokens)   | 3,403                  | CC BY-SA 4.0                                                                                 | ✔️    |
| [DanNet](#dannet)                                                                      | Wordnet                  | 66,308 (concepts) | -                      | [license](https://cst.ku.dk/projekter/dannet/license.txt)                                    | ✔️    |
| [DKHate](#dkhate)                                                                      | Hate Speech Detection    | 61,967            | 3,289                  | CC BY 4.0                                                                                    | ✔️    |
| [DaUnimorph](#daunimorph)                                                              | Morphological Inflection | 25,503            | -                      | CC BY-SA 3.0                                                                                 | ✔️    |
| [DaNED](#daned)                                                                      | Named Entity Disambiguation    | --            | train:4,626 dev:544 test:744                  | CC BY-SA 4.0                                                                                    | ✔️    |
| [DaWikiNED](#dawikined)                                                                      | Named Entity Disambiguation    | --            | 21,302                  | CC BY-SA 4.0                                                                                    | ✔️    |
| [DDisco](#ddisco)                                                                      | Discourse Coherence    | --            | -                  | CC BY-SA 4.0                                                                                    | ✔️    |


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

### DaCoref

This Danish coreference annotation contains parts of the Copenhagen Dependency Treebank  (Kromann and Lynge, 2004). It was originally annotated as part of the Copenhagen Dependency Treebank (CDT) project but never finished. This resource extends the annotation by using different mapping techniques and by augmenting with Qcodes from Wiktionary. This work is conducted by Maria Jung Barrett. Read more about it in the dedicated [DaCoref docs](dacoref_docs.md).

The dataset can be loaded with the DaNLP package:

```python
from danlp.datasets import Dacoref
dacoref = Dacoref()
# The corpus can be loaded with or without splitting into train, dev and test in a list in that order
corpus = dacoref.load_as_conllu(predefined_splits=True)
```

The dataset can also be downloaded directly:

[Download DaCoref](http://danlp-downloads.alexandra.dk/datasets/dacoref.zip) 


### DKHate

The DKHate dataset contains user-generated comments from social media platforms (Facebook and Reddit) 
annotated for various types and target of offensive language. 
The original corpus used for the [OffensEval 2020](https://sites.google.com/site/offensevalsharedtask/results-and-paper-submission) shared task can be found [here](https://figshare.com/articles/dataset/Danish_Hate_Speech_Abusive_Language_data/12220805).  
Note that only labels for the sub-task A (Offensive language identification), i.e.  `NOT` (Not Offensive) / `OFF` (Offensive), are available.

The dataset can be loaded with the DaNLP package:

```python
from danlp.datasets import DKHate
dkhate = DKHate()
test, train = dkhate.load_with_pandas()
```

The dataset can also be downloaded directly:

[Download dkhate](http://danlp-downloads.alexandra.dk/datasets/dkhate.zip)


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
Here is how you can load the dataset:

```python
from danlp.datasets import WordSim353Da

ws353 = WordSim353Da()
ws353.load_with_pandas()
```

### Danish Similarity Dataset
The [Danish Similarity Dataset](https://github.com/kuhumcst/Danish-Similarity-Dataset) 
consists of 99 word pairs annotated by 38 annotators with a similarity score (1-6).
It is constructed with frequently used Danish words.
Here is how you can load the dataset:

```python
from danlp.datasets import DSD

dsd = DSD()
dsd.load_with_pandas()
```

### Twitter Sentiment {#twitsent}

The Twitter sentiment is a small manually annotated dataset by the Alexandra Institute. It contains tags in two sentiment dimension: analytic: ['subjective' , 'objective'] and polarity: ['positive', 'neutral', 'negative' ]. It is split in train and test part. Due to Twitters privacy policy, it is only allowed to display the "tweet ID" and not the actually text. This allows people to delete their tweets. Therefore, to download the actual tweet text one need a Twitter development account and to generate the sets of login keys, read how to get started [here](https://python-twitter.readthedocs.io/en/latest/getting_started.html). Then the dataset can be loaded with the DaNLP package by setting the following environment variable for the keys:

``` TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET```

```python
from danlp.datasets import TwitterSent
twitSent = TwitterSent()

df_test, df_train = twitSent.load_with_pandas()
```

The dataset can also be downloaded directly with the labels and tweet id:

[Download TwitterSent](https://danlp.alexandra.dk/304bd159d5de/datasets/twitter.sentiment.zip) 

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

### AngryTweets

The AngryTweets sentiment dataset is a crowd-sourced dataset annotated with polarity tags: ['positive', 'neutral', 'negative' ].
The dataset contains 4122 tweets including 1727 that were annotated by one trained annotator. More annotations have been collected through the AngryTweets game resulting in 1266 tweets with double annotations. If you want to read more about the game, see the [Medium blog post](https://medium.com/danlp/angry-tweets-f%C3%B8lelser-og-annoteringer-er-p%C3%A5-spil-s%C3%A5-spil-med-eacade042c95) or the [DataTech article](https://pro.ing.dk/datatech/article/angry-tweets-vaer-med-til-bygge-datasaet-over-foelelsesladede-tweets-9496).
In the same way as the Twitter Sentiment dataset, only the ID of the tweets are made available (see [Twtitter Sentiment](#twitsent) for more details). 

Here is how to load the dataset with the DaNLP package:
```python
from danlp.datasets import AngryTweets
angrytweets = AngryTweets()

df = angrytweets.load_with_pandas()
```

The dataset (labels and tweet ids) can also be downloaded directly:

[Download AngryTweets](https://danlp-downloads.alexandra.dk/datasets/game_tweets.zip) 



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
dannet.wordnet_relations(word, eurowordnet=True)
# word ids
dannet._word_ids(word)
# synset ids
dannet._synset_ids(word)
# word from id
dannet._word_from_id(11034863)
# synset from id
dannet._synset_from_id(3514)
```


### DaUnimorph

The [UniMorph](https://unimorph.github.io/) project provides lists of word forms (for many languages) associated with their lemmas and morphological features following a universal schema which have been extracted from Wikipedia.

The Danish UniMorph is a (non-exhaustive) list of nouns and verbs. 
The following morphological features are provided : 

* the part-of-speech, i.e. noun `N` or verb `V`
* the voice (for verbs), i.e. active `ACT` or passive `PASS` 
* the mood (for verbs), i.e. infinitive `NFIN`, indicative `IND`, imperative `IMP`
* the tense (for verbs), i.e. past `PST` or present `PRS`
* the form (for nouns), i.e. indefinite `INDF` or definite `DEF`
* the case (for nouns), i.e. nominative `NOM` or genitive `GEN`
* the number (for nouns), i.e. plural `PL` or singular `SG`

For downloading DanNet through DaNLP, you can do: 

```python
from danlp.datasets import DaUnimorph

unimorph = DaUnimorph()

# you can load the database if you want to look into it by yourself
database = unimorph.load_with_pandas()
```

Once you have downloaded the DaUnimorph wrapper, you can also use the following features: 

```python

word = "trolde"
# inflections (the different forms of a word)
unimorph.get_inflections(word, pos='V', with_features=False)
""" ['troldedes', 'troldede', 'trolder', 'troldes', 'trolde', 'trold'] """
# lemmas (the root form of a word)
unimorph.get_lemmas(word, pos='N', with_features=True)
""" [{'lemma': 'trold', 'form': 'trolde', 'feats': 'N;INDF;NOM;PL', 'pos': 'N'}] """

```

### DaNED

The DaNED dataset is derived from the [DaCoref](#dacoref) (including only sentences that have at least one QID annotation) and annotated for named entity disambiguation. The dataset has been developed for DaNLP, through a Master student project, by Trong Hiêu Lâm and Martin Wu under the supervision of Maria Jung Barrett (ITU) and Ophélie Lacroix (DaNLP -- Alexandra Institute).
Each entry in the dataset is a tuple (sentence, QID) associated with a label (0 or 1) which indicate whether the entity attached to the QID is mentioned in the sentence or not. 
The same sentence occurs several times but only one of them as a label "1" because only one of the QIDs is correct.

In addition, we provide -- through the dataset -- for each QID, its corresponding knowledge graqh (KG) context extracted from Wikidata. 
For more details about the annotation process and extraction of KG context see the paper. 

The dataset can be loaded with the DaNLP package:

```python
from danlp.datasets import DaNED
daned = DaNED()
train, dev, test = daned.load_with_pandas()
```

To get the KG context (Wikidata properties and description) of a QID (from the DaNED database), you can use:

```python
qid = "Q303"
# Get Elvis Presley's Wikidata properties and description
properties, description = get_kg_context_from_qid(qid)
```

If the QID does not exist in the database, you can allow the search through Wikidata (online): 

```python
qid = "Q36620"
# Get Tycho Brahe's Wikidata properties and description
properties, description = get_kg_context_from_qid(qid, allow_online_search=True)
```



The dataset can also be downloaded directly:

[Download DaNED](http://danlp-downloads.alexandra.dk/datasets/daned.zip)



### DaWikiNED

The DaWikiNED is automatically constructed and intended to be used as a training set augmentation with the [DaNED](#daned) dataset.
The dataset has been developed for DaNLP through a student project by Trong Hiêu Lâm and Martin Wu under the supervision of Maria Jung Barrett (ITU) and Ophélie Lacroix (DaNLP -- Alexandra Institute).
Sentences come from the Danish Wikipedia. Knowledge graph contexts come from Wikidata (see [DaNED](#daned)).

The dataset can be loaded with the DaNLP package:

```python
from danlp.datasets import DaWikiNED
dawikined = DaWikiNED()
train = dawikined.load_with_pandas()
```

To get the KG context (Wikidata properties and description) of a QID (from the DaWikiNED database), you can use:

```python
qid = "Q1748"
# Get Copenhagen's Wikidata properties and description
properties, description = get_kg_context_from_qid(qid, dictionary=True)
```

If the QID does not exist in the database, you can allow the search through Wikidata (online): 

```python
qid = "Q36620"
# Get Tycho Brahe's Wikidata properties and description
properties, description = get_kg_context_from_qid(qid, allow_online_search=True)
```


The dataset can also be downloaded directly:

[Download DaWikiNED](http://danlp-downloads.alexandra.dk/datasets/dawikined.zip)


### DDisco

The DDisco dataset has been developed for DaNLP, through a Master student project.
Each entry in the dataset is annotated with a discourse coherence label (rating from 1 to 3): 

 * 1: low coherence (difficult to understand, unorganized, contained unnecessary details and can not be summarized briefly and easily)
 * 2: medium coherence
 * 3: high coherence (easy to understand, well organized, only contain details that support the main point and can be summarized briefly and easily).

Grammatical and typing errors are ignored (i.e. they do not affect the coherency score) and the coherence of a text is considered within its own domain.

The dataset can be loaded with the DaNLP package:

```python
from danlp.datasets import DDisco
ddisco = DDisco()
train, test = ddisco.load_with_pandas()
```

The dataset can also be downloaded directly:

[Download DDisco](http://danlp-downloads.alexandra.dk/datasets/ddisco.zip)



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
- Sigurbergsson, Gudbjartur Ingi  and Derczynski, Leon. [Offensive Language and Hate Speech Detection for {D}anish](https://www.aclweb.org/anthology/2020.lrec-1.430.pdf). in **LREC 2020**
