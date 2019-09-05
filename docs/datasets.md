Datasets
========
This section keeps a list of Danish NLP datasets publicly available. 

| Dataset | Task | Annotated | Size | Author/Org | License |
|---------|------|------|--------|---------|---------|
| [WordSim-353-da](https://github.com/fnielsen/dasem/tree/master/dasem/data/wordsim353-da) | Word Similarity | :heavy_check_mark: | 353 words | Finn Årup Nielsen, Original English Data belongs to [Evgeniy Gabrilovich](<http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/>) | [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/). (For the original English data) |
| [Danish Dependency Treebank](<https://github.com/UniversalDependencies/UD_Danish-DDT/tree/master>) | Part of speech tags | :heavy_check_mark: | 100k words | Annotations: PAROLE-DK project by the Danish Society for Language and Literature, Code: [github contributors ](<https://github.com/UniversalDependencies/UD_Danish-DDT/graphs/contributors>) | GNU GPL |
| [EuroParl](<http://opus.nlpl.eu/Europarl.php>) | Plain text/ Embeddings | :x: | 0.3 GB | [Statictical Machine Translation](<http://www.statmt.org/europarl/>) and [OPUS](<<http://opus.nlpl.eu/>) with the paper by J. Tiedemann, 2012, [*Parallel Data, Tools and Interfaces in OPUS.*](http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf) | From [Statictical Machine Translation](<http://www.statmt.org/europarl/>): "Not aware of any copyright restrictions of the material" |
| [OpenSubtitels2018](<http://opus.nlpl.eu/OpenSubtitles2018.php>) | Plain text/ Embeddings | :x: | 0.8GB | [Open Subtitles](<https://www.opensubtitles.org/da>) and [OPUS](<http://opus.nlpl.eu/OpenSubtitles2018.php>) with the paper  P. Lison and J. Tiedemann, 2016, [*OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles.*](http://stp.lingfil.uu.se/~joerg/paper/opensubs2016.pdf) | NONE. But please link to [Open Subtitles](<https://www.opensubtitles.org/da>) and cite the paper |
| [Wikipedia](<https://dumps.wikimedia.org/dawiki/latest/>) | Plain text/ Embeddings | :x: | 0.3GB | [Wikipedia Dumps](<https://dumps.wikimedia.org>) | NONE |



### Get started

A bash script is provided to download and extracted the plain text files. As an example if you want the European parliament corpus, run:

``` bash
bash fetch_corpus.sh --euparl

```
In the moment, the following options are supported:  `--euparl`, `--wiki` and `--opensub`. Notice that the option for Wikipedia, clones a [GitHub repository](<https://github.com/attardi/wikiextractor>) to extracted the Wikipedia dump. 
