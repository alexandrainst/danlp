Part of Speech Tagging
======================
This section is concerned with public available Part of Speech taggers in Danish. 

| Source code | Paper | Trained by | Tags | Accuracy |
|-------|-------|-------|-------|-------|
| [Polyglot](https://polyglot.readthedocs.io/en/latest/POS.html/#) | [Al-Rfou et al. (2013)](<http://www.aclweb.org/anthology/W13-3520>) | Polyglot | 12/17* |  |
| [Flair](<https://github.com/zalandoresearch/flair>) | [Akbik et. al (2018)](<https://alanakbik.github.io/papers/coling2018.pdf>) | Alexandra Instittut | 17  Universal part of speech | 97,14% |

The Danish UD treebank  uses 17 [universal part of speech tags](<https://universaldependencies.org/u/pos/index.html>):

`ADJ`: Adjective, `ADP`: Adposition , `ADV`: Adverb, `AUX`: Auxiliary verb, `CONJ`: Coordinating conjunction, `DET`: Determiner, `INTJ`: Interjection, `NOUN`: Noun, `NUM`: Numeral, `PART`: Particle `PRON`: Pronoun `PROPN`: Proper noun `PUNCT`: Punctuation `SCONJ`: Subordinating conjunction `SYM`: Symbol `VERB`: Verb `X`: Other

*The polyglot model by [Al-Rfou et al. (2013)](https://www.aclweb.org/anthology/W13-3520), is trained and 
tested on 12 universal part of speech tags originating from
[Petrov et al. (2012)](http://www.lrec-conf.org/proceedings/lrec2012/pdf/274_Paper.pdf).
The reported test accuracy is  on 96.45%. In the meantime, the 
[documentation](<https://polyglot.readthedocs.io/en/latest/POS.html>) report that the model recognizes the
17 universal part of speech tags.

![](../imgs/postag_eksempel.gif)


## Training details for Flair PoS tagger

This project provides a trained part of speech tagging model for Danish using the Flair framework from Zalando.

It is trained using the data from 
[Danish Dependency Treebank (DDT)](https://github.com/UniversalDependencies/UD_Danish-DDT/tree/master),
and by using FastText word embeddings and Flair contextual word embeddings trained in this project on data 
from Wikipedia and EuroParl corpus,
see [here](<https://github.com/alexandrainst/danlp/blob/master/docs/models/embeddings.md>).

The code for training can be found on Flairs GitHub, and the following parameters are set:
`learning_rate=1`, `mini_batch_size=32`, `max_epochs=150`, `hidden_size=256`.

The accuracy reported is from from the test set provided by DDT on a single run.
Notice, Flair report in Akbij et. al 2018 an accuracy on 97.84 ± 0.01 for the English POS tagger,
which the Danish result is rather close to.


## Get started using Part of speech tagging

Below is a small snippet for getting started with the Flair part of speech tagger trained by Alexandra Institute, but more examples can be found on [Flair](<https://github.com/zalandoresearch/flair>) GitHub page. 

```python
from danlp.models.pos_taggers import load_pos_tagger_with_flair
from flair.data import Sentence

# Load the POS tagger using the DaNLP wrapper
flair_model = load_pos_tagger_with_flair()

# Using the flair POS tagger
sentence = Sentence('jeg hopper på en bil som er rød sammen med Jens-Peter E. Hansen') 
flair_model.predict(sentence) 
print(sentence.to_tagged_string())
```

## 🎓 References 
- Rami Al-Rfou, Bryan Perozzi, and Steven Skiena. 2013. [Polyglot: Distributed Word Representations for Multilingual NLP](https://www.aclweb.org/anthology/W13-3520). In **CoNLL**.
- Alan Akbik, Duncan Blythe, and Roland Vollgraf. 2018. [Contextual String Embeddings for Sequence Labeling](https://alanakbik.github.io/papers/coling2018.pdf). In **COLING**.
- Slav Petrov, Dipanjan Das, and Ryan McDonald. 2012 [A Universal Part-Of-Speech Tagset](http://www.lrec-conf.org/proceedings/lrec2012/pdf/274_Paper.pdf)." In **LREC**.