Datasets
========
This section keeps a list of Danish NLP datasets publicly available. 

| Dataset | Task | Words | Sents | License |
|---------|-------------|------|--------|---------|
| [OpenSubtitles2018](<http://opus.nlpl.eu/OpenSubtitles2018.php>) | Translation | 206,700,000 | 30,178,452 |[None](http://opus.nlpl.eu/OpenSubtitles2018.php) |
| [EU Bookshop](http://opus.nlpl.eu/EUbookshop-v2.php) | Translation | 208,175,843 | 8,650,537 | - |
| [EuroParl7](http://opus.nlpl.eu/Europarl.php) | Translation | 47,761,381 | 2,323,099	 | [None](http://www.statmt.org/europarl/)|
| [ParaCrawl5](https://paracrawl.eu/) | Translation | - | - | [CC0](https://paracrawl.eu/releases.html)
| WikiANN | NER | 832.901 | 95.924 |[ODC-BY 1.0](http://nlp.cs.rpi.edu/wikiann/)|
| [Danish Dependency Treebank](<https://github.com/UniversalDependencies/UD_Danish-DDT/tree/master>) | POS, NER |  100,733 |  5,512 | [CC BY-SA 4.0](https://github.com/UniversalDependencies/UD_Danish-DDT/blob/master/README.md) |
| [Wikipedia](<https://dumps.wikimedia.org/dawiki/latest/>) | Raw | 0.3GB* | - | [CC BY-SA 3.0](https://dumps.wikimedia.org/legal.html) |
| [WordSim-353-da](https://github.com/fnielsen/dasem/tree/master/dasem/data/wordsim353-da) | Word Similarity  | 353 | - | [CC BY 4.0](https://github.com/fnielsen/dasem/blob/master/dasem/data/wordsim353-da/LICENSE)| 

### Get started

A bash script is provided to download and extracted the plain text files. As an example if you want the European parliament corpus, run:

``` bash
bash fetch_corpus.sh --euparl
```
In the moment, the following options are supported:  `--euparl`, `--wiki` and `--opensub`. Notice that the option for Wikipedia, clones a [GitHub repository](<https://github.com/attardi/wikiextractor>) to extracted the Wikipedia dump. 


## 🎓 References
- Lev Finkelstein, Evgeniy Gabrilovich, Yossi Matias, Ehud Rivlin, Zach Solan, Gadi Wolfman, and Eytan Ruppin. 2002. [Placing Search in Context: The Concept Revisited](http://www.cs.technion.ac.il/~gabr/papers/tois_context.pdf). In  **ACM TOIS**.