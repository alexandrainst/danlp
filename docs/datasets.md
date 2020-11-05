Datasets
========

This section keeps a list of Danish NLP datasets publicly available. 

| Dataset | Task | Words | Sents | License | DaNLP |
|---------|------|-------|-------|---------|-----------------|
| [OpenSubtitles2018](<http://opus.nlpl.eu/OpenSubtitles2018.php>) | Translation | 206,700,000 | 30,178,452 |[None](http://opus.nlpl.eu/OpenSubtitles2018.php) | ❌ |
| [EU Bookshop](http://opus.nlpl.eu/EUbookshop-v2.php) | Translation | 208,175,843 | 8,650,537 | - | ❌ |
| [Europarl7](http://www.statmt.org/europarl/) | Translation | 47,761,381 | 2,323,099	 | [None](http://www.statmt.org/europarl/) | ❌ |
| [ParaCrawl5](https://paracrawl.eu/) | Translation | - | - | [CC0](https://paracrawl.eu/releases.html) | ❌ |
| [WikiANN](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#wikiann)| NER | 832.901 | 95.924 |[ODC-BY 1.0](http://nlp.cs.rpi.edu/wikiann/)| ✔️ |
| [UD-DDT (DaNE)](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane) | DEP, POS, NER |  100,733 |  5,512 | [CC BY-SA 4.0](https://github.com/UniversalDependencies/UD_Danish-DDT/blob/master/README.md) | ✔️ |
| [LCC Sentiment](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#lcc-sentiment) | Sentiment | 10.588 | 499 | [CC BY](https://github.com/fnielsen/lcc-sentiment/blob/master/LICENSE) | ✔️ |
| [Europarl Sentiment1](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#europarl-sentiment1) | Sentiment | 3.359 | 184 | None | ✔️ |
| [Europarl Sentiment2](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#europarl-sentiment2) | sentiment |  | 957 | CC BY-SA 4.0 | ✔️ |
| [Wikipedia](https://dumps.wikimedia.org/dawiki/latest/) | Raw | - | - | [CC BY-SA 3.0](https://dumps.wikimedia.org/legal.html) | ❌ |
| [WordSim-353](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#wordsim-353) | Word Similarity  | 353 | - | [CC BY 4.0](https://github.com/fnielsen/dasem/blob/master/dasem/data/wordsim353-da/LICENSE)| ✔️ |
| [Danish Similarity Dataset](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-similarity-dataset) | Word Similarity  | 99 | - | [CC BY 4.0](https://github.com/fnielsen/dasem/blob/master/dasem/data/wordsim353-da/LICENSE)| ✔️ |
| [Twitter Sentiment](https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#Twitter-Sentiment) | Sentiment | - | train: 1215, test: 512 | Twitter privacy policy applies | ✔️ |

It is also recommend to check out Finn Årup Nielsen's [dasem github](https://github.com/fnielsen/dasem) which also provides script for loading different Danish corpus. 

### Danish Dependency Treebank (DaNE)

The Danish UD treebank (Johannsen et al., 2015, UD-DDT) is a
conversion of the Danish Dependency Treebank (Buch-Kromann et
al. 2003) based on texts from Parole (Britt, 1998).
UD-DDT has annotations for dependency parsing and POS. 
The dataset was annotated with Named Entities for **PER**, **ORG** and **LOC** 
by the Alexandra Institute in the DaNE dataset (Hvingelby et al. 2020).
To read more about how the dataset was annotated with POS and DEP tags we refer to the
[Universal Dependencies](https://github.com/UniversalDependencies/UD_Danish-DDT/blob/master/README.md) page.
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
The WordSim-353 dataset [(Finkelstein et al. 2002)](http://www.cs.technion.ac.il/~gabr/papers/tois_context.pdf) 
contains word pairs annotated with a similarity score (1-10). It is common to use it to do intrinsic evaluations 
on word embeddings to test for syntactic or semantic relationships between words. The dataset has been 
[translated to Danish](https://github.com/fnielsen/dasem/tree/master/dasem/data/wordsim353-da) by Finn Årup Nielsen.

### Danish Similarity Dataset
The [Danish Similarity Dataset](https://github.com/kuhumcst/Danish-Similarity-Dataset) 
consists of 99 word pairs annotated by 38 annotators with a similarity score (1-6).
It is constructed with frequently used Danish words.

### Twitter Sentiment

The Twitter sentiment is a small manually annotated dataset by the Alexandra Institute. It contains tags in two sentiment dimension: analytic: ['subjective' , 'objective'] and polarity: ['positive', 'neutral', 'negative' ]. It is split in train and test part. Due to Twitters privacy policy, it is only allowed to display the "tweet ID" and not the actually text. This allows people to delete their tweets. Therefore, to download the actual tweet text one need a Twitter development account and to generate the sets of login keys, read how to get started [here](https://python-twitter.readthedocs.io/en/latest/getting_started.html). Then the dataset can be loaded with the DaNLP package by setting the following environment variable for the keys:

``` TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET```|

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


## 🎓 References
- Johannsen, Anders, Martínez Alonso, Héctor and Plank, Barbara. “Universal Dependencies for Danish”. TLT14, 2015.
- Keson, Britt (1998). Documentation of The Danish Morpho-syntactically Tagged PAROLE Corpus. Technical report, DSL
- Matthias T. Buch-Kromann, Line Mikkelsen, and Stine Kern Lynge. 2003. "Danish dependency treebank". In **TLT**.
- Rasmus Hvingelby, Amalie B. Pauli, Maria Barrett, Christina Rosted, Lasse M. Lidegaard and Anders Søgaard. 2020. DaNE: A Named Entity Resource for Danish. In **LREC**.
- Xiaoman Pan, Boliang Zhang, Jonathan May, Joel Nothman, Kevin Knight and Heng Ji. 2017. [Cross-lingual Name Tagging and Linking for 282 Languages](https://aclweb.org/anthology/P17-1178). In **ACL**.
- Lev Finkelstein, Evgeniy Gabrilovich, Yossi Matias, Ehud Rivlin, Zach Solan, Gadi Wolfman, and Eytan Ruppin. 2002. [Placing Search in Context: The Concept Revisited](http://www.cs.technion.ac.il/~gabr/papers/tois_context.pdf). In  **ACM TOIS**.
- Uwe Quasthoff, Matthias Richter and Christian Biemann. 2006. [Corpus Portal for Search in Monolingual Corpora](https://www.aclweb.org/anthology/L06-1396/). In **LREC**.
