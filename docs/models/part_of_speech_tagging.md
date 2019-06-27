Part of Speech Tagging
===============
![](../imgs/postag_eksempel.gif)

This section is concerned with open source POS taggers available in Danish. 

| Source code | Paper | Trained by          | Reported Accuracy |
|-------|-------|-------|-------|
| [Polyglot](https://polyglot.readthedocs.io/en/latest/POS.html/#) | [Al-Rfou et al. (2013)](<http://www.aclweb.org/anthology/W13-3520>) | Polyglot | 96,4%* |
| [Flair](<https://github.com/zalandoresearch/flair>) | [Akbik et. al (2018)](<https://alanakbik.github.io/papers/coling2018.pdf>) | Alexandra Instittut | 97,14% |

*The reported accuracy form the Polyglot model is taken from the paper Akbik Al-Rfou et al. 2013. 

The Danish UD treebank  uses 17 universal part of speech tags:
`ADJ`: Adjective   `ADP`: Adposition `ADV`: Adverb `AUX`: Auxiliary verb `CONJ`: Coordinating conjunction `DET`: determiner `INTJ`: interjection `NOUN`: Noun `NUM`: Numeral `PART`: Particle `PRON`: Pronoun `PROPN`: Proper noun `PUNCT`: Punctuation `SCONJ`: Subordinating conjunction `SYM`: Symbol `VERB`: Verb `X`: Other

Part of speech tagging has several applications. It can for example be used as surface features, indicating how many word of each tags are present in the text. Knowing the tags is also relevant if you want to find the lemma of a word. An idea for use could also concern text augmentation, where you want to make small permutation to you text. This could be done by finding a token with a certain part of speech tag, eg. noun, and then find and shift a similar noun using e.g a word2vec model. In that way the structure of the sentence is preserved. 


## Get started using Part of speech tagging

Download the flair part of speech tagger model using the script in the parent 'models' folder. Below is a small snippet for getting started with the Flair part of speech tagging, but more examples can be found on [Flair](<https://github.com/zalandoresearch/flair>) GitHub page. 

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

The Jupyter notebook `POS_tagger_exampel.ipynb` ([here](<https://github.com/alexandrainst/danlp/blob/master/examples/POS_tagger_exampel.ipynb>)) provides some more getting started examples for both the Polyglot model and the Flair model.

## Training details for Flair Pos tagger

This project provides a trained part of speech tagging model for Danish using the Flair framework from Zalendo.

It is trained using the data from  [Danish UD treebank  ](<https://github.com/UniversalDependencies/UD_Danish-DDT/tree/master>), and by using FastText word embeddings and Flair contextual word embeddings trained in this project on data from Wikipedia and EuroParl corpus, see [here](<https://github.com/alexandrainst/danlp/blob/master/docs/models/embeddings.md>) .

The code for training can be found on Flairs GitHub, and the following parameters are set:              learning_rate=1, mini_batch_size=32, max_epochs=150, hidden_size=256.

The accuracy reported is from from the test set provided by Danish Dependecy Treebank on a single run. Notice,  Flair report in Akbij et. al 2018 an accuracy on 97,84 +/- 0,01 for the English POS tagger, which the Danish result is rather close to.


## References 

Al-Rfou et al. (2013). [Polyglot: DistributedWordRepresentationsforMultilingualNLP](https://www.aclweb.org/anthology/W13-3520). Proceedings of the Seventeenth Conference on Computational Natural Language Learning

Akbik et. al (2018), [ContextualStringEmbeddingsforSequenceLabeling](https://alanakbik.github.io/papers/coling2018.pdf). Proceedings of the 27th International Conference on Computational Linguistics.

