Named Entity Recognition
===============
This repository keeps a list of pretrained NER models publicly available in Danish.

| Model | Paper | Trained by | Tags |
|------|-------|------------|------|
| [Polyglot](https://polyglot.readthedocs.io/en/latest/POS.html/#) | [Al-Rfou et al. (2014)](https://arxiv.org/abs/1410.3791) | Polyglot | PER, ORG, LOC|
| [daner](https://github.com/ITUnlp/daner) | [Derczynski et al. (2014)](https://www.aclweb.org/anthology/E14-2016) | [ITU NLP](https://nlp.itu.dk/) | PER, ORG, LOC |
| Multilingual BERT |  | [MIPT](https://mipt.ru/english/) |



## 📈 Benchmarks

The benchmarks has been performed on the test part of the
[Danish Dependency Treebank](https://github.com/alexandrainst/danlp/blob/add-ner/docs/datasets.md#danish-dependency-treebank).
The treebank is annotated by the Alexandra Institute with the **LOC**, **ORG** and **PER** entity tags. Below is the achieved F1 score on the test set:


| Model | LOC | ORG | PER | AVG |
|-------|-----|-----|-----|-----|
| Multilingual BERT | 78.49 | 73.23 | 89.39 | **80.37** |
| daner | 61.38 | 27.55 | 70.05 | 52.99 |
| Polyglot | 58.33 | 25.40 | 20.69 | 34.81 |

## 🎓 References
- Rami Al-Rfou, Vivek Kulkarni, Bryan Perozzi and Steven Skiena. 2015. [POLYGLOT-NER: Massive Multilingual Named Entity Recognition](https://arxiv.org/abs/1410.3791). In **SDM**.
- Leon Derczynski, Camilla V. Field and Kenneth S. Bøgh. 2014. [DKIE: Open Source Information Extraction for Danish](https://www.aclweb.org/anthology/E14-2016). In **EACL**.