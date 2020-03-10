Sentiment Analysis
============================

Sentiment analysis refers to identifying an emotion or opinion in a text. The following focus on the polarity of a sentence meaning the  tone of positive, neutral or negative. 

In Danish there is so-far non open source annotated training set. 

## Models available in Danish 

| Model                                              | Data     | Licence                    | Trained by               | Tags                                                         |
| -------------------------------------------------- | -------- | -------------------------- | ------------------------ | ------------------------------------------------------------ |
| [AFINN](https://github.com/fnielsen/afinn)         | Wordlist | BSD                        | Finn Årup Nielsen        | score (integers), {minus: negative, zero: neutral, plus: positive} |
| [SentidaV2](<https://github.com/esbenkc/emma>)[^1] | Wordlist | (**!**) Non commercial[^2] | Søren Orm and Esben Kran | score (real numbers) {minus: negative, zero: neutral, plus: positive} |

[1] : The sentida tool have a first version in R: <https://github.com/Guscode/Sentida>

[2] : The documentation requires to contact the authors

#### Zero-shot Cross-lingual transfer example

An example of utilizing an dataset in another language to be able to make predicts on Danish without seeing Danish training data is shown in this 
[notebok](<https://github.com/alexandrainst/danlp/blob/sentiment-start/examples/Zero_shot_sentiment_analysi_example.ipynb>). It is trained on English movie reviews from IMDB, and
it uses multilingual embeddings from [Artetxe et al. 2019](https://arxiv.org/pdf/1812.10464.pdf) called 
[LASER](<https://github.com/facebookresearch/LASER>)(Language-Agnostic SEntence Representations).



## Evaluation dataset

 The exist a small set of manual annotated evaluation dataset for sentiment analysis.

| Name                                                         | Created by          | Size      | Licence      | Tags                                                      | Domain                                      | DaNLP |
| ------------------------------------------------------------ | ------------------- | --------- | ------------ | --------------------------------------------------------- | ------------------------------------------- | ----- |
| Twitter_sentiment                                            | Alexandra Institute | Test: 400 | BSD 3-Clause | [positive, neutral, negative] and [objective, subjective] | SoMe (twiiter)                              |       |
| [Europarl_sentiment](<https://github.com/fnielsen/europarl-da-sentiment>) | Finn Årup Nielsen   | Test: 184 | open         | integer score: -5 to 5                                    | EuroParl corpus                             |       |
| [lcc-_sentiment](<https://github.com/fnielsen/lcc-sentiment>) | Finn Årup Nielsen   | Test: 499 | CC           | integer score: -5 to 5                                    | Web , news (The Leipzig Corpora Collection) |       |



## 📈 Benchmarks 

The benchmark is made by  grouping the relevant models scores and relevant datasets scores into the there classes defined as follows: a score of zero to be neutral, a positive score to be positive and a negative score to be negative. 

| **Model** / dataset | Twitter_sentiment                                            | Europarl_sentiment                                           | lcc-_sentiment                                               |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Afinn               | **Accuracy**:  0.5  **F1-scores**: Negative: 0.58,  Neutral: 0.31 Positive: 0.51 | **Accuracy**:  0.68  **F1-scores**: Negative: 0.62,  Neutral: 0.71 Positive: 0.70 | **Accuracy**:  0.66  **F1-scores**: Negative: 0.47,  Neutral: 0.73 Positive: 0.62 |
| SentidaV2           | **Accuracy**:  0.51  **F1-scores**: Negative: 0.62,  Neutral:  0.0, Positive: 0.51 | **Accuracy**:  0.49  **F1-scores**: Negative: 0.56,  Neutral: 0.25, Positive: 0.58 | **Accuracy**:  0.38  **F1-scores**: Negative: 0.51,  Neutral: 0.08, Positive: 0.49 |
|                     |                                                              |                                                              |                                                              |



## 🎓 References 
- Mikel Artetxe, and Holger Schwenk. 2019. 
  [Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond.](https://arxiv.org/pdf/1812.10464.pdf). 
  In **TACL**.
  
  

 
