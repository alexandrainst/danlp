"""
Evaluation script for sentiment analyis
The script benchmark on the following dataset where scores are converted into a three class problem: positiv, neutral, negative:
    - Europarl_sentiment
    - Lcc_sentiment 
The script benchmark the following models where scores are converted into a three class problem:
    - BERT Tone for positiv, negative, neutral 
            the model is integrated in danlp package 
    - Afinn:
           Requirements:
               - pip install afinn
    - Sentida:
           Sentida is converted to three class probelm by fitting a treshold for neutral on manualt annotated twitter corpus.
           The script downloadsfilles from sentida github and place them in cache folder
           Requirement:
               - pip install sentida==0.5.0
 
"""

from danlp.datasets import EuroparlSentiment1, LccSentiment
from danlp.models import load_bert_tone_model, load_spacy_model
from afinn import Afinn
import operator
import time
from utils import *


def afinn_benchmark(datasets):
    afinn = Afinn(language='da', emoticons=True)
    
    for dataset in datasets:
        if dataset == 'euparlsent':
            data = EuroparlSentiment1()
        if dataset == 'lccsent':
            data = LccSentiment()

        df = data.load_with_pandas()


        start = time.time()
        df['pred'] = df.text.map(afinn.score).map(sentiment_score_to_label)
        print_speed_performance(start, len(df))
        df['valence'] = df['valence'].map(sentiment_score_to_label)

        f1_report(df['valence'], df['pred'], 'Afinn', dataset)
        
        
def sentida_benchmark(datasets):

    from sentida import Sentida
    sentida =  Sentida()
    
    def sentida_score(sent):
        return sentida.sentida(sent, output ='total')
    
    for dataset in datasets:
        if dataset == 'euparlsent':
            data = EuroparlSentiment1()
        if dataset == 'lccsent':
            data = LccSentiment()

        df = data.load_with_pandas()


        start = time.time()
        df['pred'] = df.text.map(sentida_score).map(sentiment_score_to_label_sentida)
        print_speed_performance(start, len(df))
        df['valence'] = df['valence'].map(sentiment_score_to_label)


        f1_report(df['valence'], df['pred'], 'Sentida', dataset)

def senda_benchmark(datasets):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    tokenizer = AutoTokenizer.from_pretrained("pin/senda")
    model = AutoModelForSequenceClassification.from_pretrained("pin/senda")

    # create 'senda' sentiment analysis pipeline 
    senda_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    for dataset in datasets:
        if dataset == 'euparlsent':
            data = EuroparlSentiment1()
        if dataset == 'lccsent':
            data = LccSentiment()

        df = data.load_with_pandas()

        df['valence'] = df['valence'].map(sentiment_score_to_label)
        start = time.time()
        df['pred'] = df.text.map(lambda x: senda_pipeline(x)[0]['label'])
        print_speed_performance(start, len(df))

        f1_report(df['valence'], df['pred'], 'Senda', dataset)

def bert_sent_benchmark(datasets):
    model = load_bert_tone_model()
    
    for dataset in datasets:
        if dataset == 'euparlsent':
            data = EuroparlSentiment1()
        if dataset == 'lccsent':
            data = LccSentiment()

        df = data.load_with_pandas()


        df['valence'] = df['valence'].map(sentiment_score_to_label)
        # predict with bert sentiment 
        start = time.time()
        df['pred'] = df.text.map(lambda x: model.predict(x, analytic=False)['polarity'])
        print_speed_performance(start, len(df))
        spellings_map = {'subjective': 'subjektivt', 'objective': 'objektivt', 'positive': 'positiv', 'negative': 'negativ', 'neutral': 'neutral'}
        df['pred'] = df['pred'].map(lambda x: spellings_map[x])

        f1_report(df['valence'], df['pred'], 'BERT_Tone (polarity)', dataset)

def spacy_sent_benchmark(datasets):
    
    nlpS = load_spacy_model(textcat='sentiment', vectorError=True)
   
    for dataset in datasets:
        if dataset == 'euparlsent':
            data = EuroparlSentiment1()
        if dataset == 'lccsent':
            data = LccSentiment()

        df = data.load_with_pandas()
        
        df['valence'] = df['valence'].map(sentiment_score_to_label)
        
        # predict with spacy sentiment 
        def predict(x):
            doc = nlpS(x)
            pred = max(doc.cats.items(), key=operator.itemgetter(1))[0]
            #match the labels 
            labels = {'positiv': 'positive', 'neutral': 'neutral', 'negativ': 'negative'}
            return labels[pred]

        spellings_map = {'subjective': 'subjektivt', 'objective': 'objektivt', 'positive': 'positiv', 'negative': 'negativ', 'neutral': 'neutral'}
        start = time.time()
        df['pred'] = df.text.map(lambda x: spellings_map[predict(x)])
        print_speed_performance(start, len(df))

        f1_report(df['valence'], df['pred'], 'Spacy sentiment (polarity)', dataset)

if __name__ == '__main__':
    sentida_benchmark(['euparlsent','lccsent'])
    afinn_benchmark(['euparlsent','lccsent'])
    bert_sent_benchmark(['euparlsent','lccsent'])
    spacy_sent_benchmark(['euparlsent','lccsent'])
    senda_benchmark(['euparlsent','lccsent'])