Pretrained Danish embeddings
============================
This repository keeps a list of pretrained word embeddings publicly available in Danish. The `download_embeddings.py`
and `load_embeddings.py` provides functions for downloading the wordembeddings as well as prepare them for use in 
popular NLP frameworks.

| Name | Model | Data | Vocab | Unit | Task  | Pretrainer |
|------|-------|------|:-----:|------|-------|---------------|
| [connl.da.wv](http://vectors.nlpl.eu/repository/#) | word2vec | [Danish CoNLL17](http://universaldependencies.org/conll17/) | 1.655.886 | Word | Skipgram | [University of Oslo](https://www.mn.uio.no/ifi/english/) |
| [news.da.wv](https://loar.kb.dk/handle/1902/329) | word2vec | OCR Newspapers 1880-2005 | 2.404.836 | Word | Skipgram | [Det Kgl. Bibliotek](http://www.kb.dk) |
| [cc.da.swv](https://fasttext.cc/docs/en/crawl-vectors.html) | fastText | CC + Wiki | 2.000.000 | Char N-gram | Skipgram |  [Facebook AI Research](https://research.fb.com/category/facebook-ai-research/) |
| [wiki.da.swv](https://fasttext.cc/docs/en/pretrained-vectors.html)| fastText | Wikipedia | 312.956 | Char N-gram | Skipgram | [Facebook AI Research](https://research.fb.com/category/facebook-ai-research/) |
| forward_embedding backward_embedding | Flair | Wikipedia + Europarl | | Char | LM | [Alexandra Institute](https://alexandra.dk/uk) |

Embeddings is a way of representing text as vectors of floats, and can be calculated both for words -, sentences - or documents embeddings. There exist different models for training embeddings, and roughly it can be deviated into static and dynamic embeddings. The static is providing a look up of the vector representation for each word in the vocabulary e.g a word2vec model. Another example is the fastText embeddings which uses n-gram characters as units, and therefor is robust to e.g misspellings.   The dynamic embeddings is contextual in the sense that the embeddings of each word is dependent on the sentence they appear in. In that way homonyms get different vector representations. An example of this is the Flair model where the embeddings is taken from a language model learning to predict the next character in a sentence.  But a lot of ways and models exist to create such embeddings, and it is important to think of how the embeddings is trained and of what data it is trained on. For example if a bias (e.g gender bias) occur in the data it will be present in the embeddings as well.  

## Using word embeddings for analysis

Word embeddings are essentially a representation of a word in a n-dimensional space. Having a vector representation of a word enables us to find distances between words. In `load_embeddings.py` we have provided functions to download pretrained word embeddings and load them with the two popular NLP frameworks [spaCy](https://spacy.io/) and [Gensim](https://radimrehurek.com/gensim/).

This snippet shows how to automatically download and load pretrained word embeddings e.g. trained on the CoNLL17 dataset.
```python
from load_embeddings import load_wv_with_gensim, load_wv_with_spacy

# Load with gensim
word_embeddings = load_wv_with_gensim('connl.da.wv')

word_embeddings.most_similar(positive=['københavn', 'england'], negative=['danmark'], topn=1)
# [('london', 0.7156291604042053)]

word_embeddings.doesnt_match("vand sodavand brød vin juice".split())
# 'brød'

word_embeddings.similarity('københavn', 'århus')
# 0.550142

word_embeddings.similarity('københavn', 'esbjerg')
# 0.48161203


# Load with spacy
word_embeddings = load_wv_with_spacy('connl.da.wv')

```

#### Flair contextual embeddings and Flair framework

##### Training details

This repository provides Flair word embeddings trained on Danish data from Wikipedia and EuroParl both forwards and backwards. Have a look at Flairs own [GitHub](<https://github.com/zalandoresearch/flair>) page to get the code for how it is trained. The hyperparameter are set as follows: hidden_size=1032, nlayers=1, sequence_length=250,  mini_batch_size=50, max_epochs=5

##### Example of use

 The [GitHub](<https://github.com/zalandoresearch/flair>)  page for Flair also provides nice tutorials and an easy framework for using other word embeddings as well and concatenate them. In the snippet below you can see how to load the pretrained Danish embeddings and an example of simple use. 

```Python
from danlp.models.embeddings import load_context_embeddings_with_flair
from flair.data import Sentence

# Use the wrapper from DaNLP to download and load embeddings with Flair
stacked_embeddings = load_context_embeddings_with_flair()

# Embed two different sentences
sentence1 = Sentence('Han fik bank')
sentence2 = Sentence('Han fik en ny bank')
stacked_embeddings.embed(sentence1)
stacked_embeddings.embed(sentence1)

# Show that it is contextual in the sense 'bank' has different embedding after context
print('{} entence out of {} is equal'.format(int(sum(sentence2[4].embedding==sentence1[2].embedding)), len(sentence1[2].embedding)))
# 52 ud af 2364

```


The trained Flair word embeddings has been used in training the Danish Part of speech model with Flair, check it out [here](<https://github.com/alexandrainst/danlp/blob/master/docs/models/part_of_speech_tagging.md>). 



## 📈 Benchmarks

To evaluate word embeddings it is common to do intrinsic evaluations to directly test for syntactic or 
semantic relationships between words. The WordSimilarity-353 [4] dataset contains word pairs 
annotated with a similarity score (1-10) and calculating the correlation between the word embedding similarity and the
similarity score gives an indication of how well the word embeddings captures relationships between words. The dataset 
has been [translated to Danish](https://github.com/fnielsen/dasem/tree/master/dasem/data/wordsim353-da) by Finn Aarup Nielsen. 

| Model | Spearman's rho | OOV |
| ------:|:----------------:|:----------:|
| cc.da.wv | **0.5917** | 0% |
| wiki.da.wv | 0.5851 | 5.01% |
| connl.da.wv | 0.5243 | 5.01% |
| news.da.wv | 0.4961 | 5.6% |

## References

[1] Mikolov et al. (2013). [Distributed Representations of Words and Phrasesand their Compositionality](). NeurIPS'13

[2] Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov (2016), [Enriching Word Vectors with Subword Information](). ACL

[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)

[4] Finkelstein et al. [Placing search in context: The con-cept revisited]()


