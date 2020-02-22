Named Entity Recognition
===============
This repository keeps a list of pretrained NER models publicly available in Danish.

| Model | Train Data | Trained by | Tags | DaNLP |
|-------|-------|------------|------|-------|
| [Flair](<https://github.com/zalandoresearch/flair>) | DaNE | The Alexandra Institute | PER, ORG, LOC | ✔️ |
| Multilingual-BERT | DaNE+NorNE | The Alexandra Institute | PER, ORG, LOC | ✔️ |
| Danish-BERT | DaNE | The Alexandra Institute | PER, ORG, LOC | ✔️ |
| [Polyglot](https://polyglot.readthedocs.io/en/latest/POS.html/#) | Wikipedia | Polyglot | PER, ORG, LOC | ❌ | 
| [daner](https://github.com/ITUnlp/daner) | [Derczynski et al. (2014)](https://www.aclweb.org/anthology/E14-2016) | [ITU NLP](https://nlp.itu.dk/) | PER, ORG, LOC | ❌ |

#### Multilingual-BERT
Google released the Multilingual BERT. We have.

You can download the weights for the finetuned NER model directly here:
Or you can load the model through the DaNLP as a pytorch model in the transformers framework.

#### Flair
The Flair model 

#### Danish Bert
The NER model is trained on top of the Danish BERT provided by BotXO.

## 📈 Benchmarks

The benchmarks has been performed on the test part of the
[DaNE](https://github.com/alexandrainst/danlp/blob/add-ner/docs/datasets.md#danish-dependency-treebank) dataset.
We are only reporting the scores on the `LOC`, `ORG` and `PER` entities as the `MISC` category has limited 
practical use.
The table below has the achieved F1 score on the test set:


| Model | LOC | ORG | PER | AVG |
|-------|-----|-----|-----|-----|
| Multilingual BERT | 78.49 | 73.23 | 89.39 | **80.37** |
| flair | 86.02 | 61.61 | 93.11 | 80.24 |
| daner | 61.38 | 27.55 | 70.05 | 52.99 |
| Polyglot | 58.33 | 25.40 | 20.69 | 34.81 |



## :hatching_chick: Get started using Name Entity Recognition 

Below is a small snippet for getting started with the Flair name entity recognition tagger trained by Alexandra Institute. More examples can be found on [Flair](<https://github.com/zalandoresearch/flair>) GitHub page, and the NER tagger is also integrated direct in the flair framework.

```python
from danlp.models.ner_taggers import load_ner_tagger_with_flair
from flair.data import Sentence

# Load the NER tagger using the DaNLP wrapper
flair_model = load_ner_tagger_with_flair()

# Using the flair NER tagger
sentence = Sentence('jeg hopper på en bil som er rød sammen med Jens-Peter E. Hansen') 
flair_model.predict(sentence) 
print(sentence.to_tagged_string())
```



## :wrench: Training details Flair NER tagger

This project has provided a name entity tagged version of [Danish Dependency Treebank (DDT)](https://github.com/UniversalDependencies/UD_Danish-DDT/tree/master) which enables the training of name entity recognitions models for Danish. As a first step of developing new NER models on Danish we provide a model trained using the Flair framework from Zalando.

It is trained using the data from DDT, and by using FastText word embeddings and Flair contextual word embeddings trained in this project on data rom Wikipedia and EuroParl corpus, see [here](<https://github.com/alexandrainst/danlp/blob/master/docs/models/embeddings.md>).

The code for training can be found on Flairs GitHub, and the following parameters are set:
`learning_rate=1`, `mini_batch_size=32`, `max_epochs=150`, `hidden_size=256`.

The accuracy reported is from from the test set provided by DDT on a single run.





## 🎓 References

- Rami Al-Rfou, Vivek Kulkarni, Bryan Perozzi and Steven Skiena. 2015. [POLYGLOT-NER: Massive Multilingual Named Entity Recognition](https://arxiv.org/abs/1410.3791). In **SDM**.
- Leon Derczynski, Camilla V. Field and Kenneth S. Bøgh. 2014. [DKIE: Open Source Information Extraction for Danish](https://www.aclweb.org/anthology/E14-2016). In **EACL**.