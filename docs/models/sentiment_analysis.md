Sentiment Analysis
============================


## Models available in Danish 

| Model                                                        | Data           | Licence | Trained by | Tags | DaNLP |
| ------------------------------------------------------------ | -------------- | ------- | ---------- | ---- | ----- |
| [AFINN](https://github.com/fnielsen/afinn/blob/master/LICENSE) | Wordlist       |         |            |      |       |
| Sentida                                                      | Wordlist       |         |            |      |       |
| laser_imdb*                                                  | IMDB (English) |         |            |      |       |
|                                                              |                |         |            |      |       |
|                                                              |                |         |            |      |       |

##### *zero-shot Cross-lingual transfer example
An example of utilizing an dataset in another language to be able to make predicts on Danish without seeing Danish training data is shown in this 
[notebok](<https://github.com/alexandrainst/danlp/blob/sentiment-start/examples/Zero_shot_sentiment_analysi_example.ipynb>). It is trained on English movie reviews from IMDB, and
it uses multilingual embeddings from [Artetxe et al. 2019](https://arxiv.org/pdf/1812.10464.pdf) called 
[LASER](<https://github.com/facebookresearch/LASER>)(Language-Agnostic SEntence Representations).



## Annotated dataset

 

| Name              | Created by          | Size                | Tags                                                      | Domain  |
| ----------------- | ------------------- | ------------------- | --------------------------------------------------------- | ------- |
| Twitter_sentiment | Alexandra Institute | Test: 400, dev: (?) | [positive, neutral, negative] and [objective, subjective] | Twitter |
|                   |                     |                     |                                                           |         |
|                   |                     |                     |                                                           |         |



## 📈 Benchmarks 

| **Model** |      |      |
| --------- | ---- | ---- |
|           |      |      |
|           |      |      |
|           |      |      |



## 🎓 References 
- Mikel Artetxe, and Holger Schwenk. 2019. 
  [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond.](https://arxiv.org/pdf/1812.10464.pdf). 
  In **TACL**.
  
  

 
