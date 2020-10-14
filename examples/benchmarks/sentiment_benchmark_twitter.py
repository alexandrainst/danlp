"""
Evaluation script for sentiment analyis on TWITTER DATA
This script requires an acount for TWITTER DEVLOPMENT API and that following the keys are set as envoriment variable:
TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET|


The script test both polarity (positive, negative and neutral) and analytic (objective, subjective)

The script benchmark on the following dataset where scores are converted into a three class problem: positiv, neutral, negative:
    - Europarl_sentiment
    - Lcc_sentiment 

The script benchmark the following models where scores are converted into a three class problem:
    - BERT Tone for positiv, negative, neutral 
            the model is integrated in danlp package 
    - Afinn:
           Requirements:
               - pip install afinn
    - SentidaV2:
           Sentida is converted to three class probelm by fitting a treshold for neutral on manualt annotated twitter corpus.
           The script downloadsfilles from sentida github and place them in cache folder
           Requirement:
               - pip install sentida==0.5.0
                       
 
"""

from danlp.datasets import TwitterSent
from danlp.models import load_bert_tone_model, load_spacy_model
from danlp.metrics import f1_report
import os
import urllib
from pathlib import Path
import sys
import operator
import time
from .utils import print_speed_performance

## Load the Twitter data
twitSent = TwitterSent()
df_val, _ = twitSent.load_with_pandas()

def to_label(score):
    if score == 0:
        return 'neutral'
    if score < 0:
        return 'negativ'
    else:
        return 'positiv'

def to_label_sentida(score):
    # the threshold of 0.4 is fitted on a manually annotated twitter corpus for sentiment on 1327 examples
    if score > 0.4:
        return 'positiv'
    if score < -0.4:
        return 'negativ'
    else:
        return 'neutral'
    
    
def afinn_benchmark():
    from afinn import Afinn
    afinn = Afinn(language='da', emoticons=True)
    start = time.time()
    df_val['afinn'] = df_val.text.map(afinn.score).map(to_label)
    print_speed_performance(start, len(df_val))

    f1_report(df_val['polarity'], df_val['afinn'], 'Afinn', "twitter_sentiment(val)")


def sentida_benchmark():

    from sentida import Sentida
    sentida =  Sentida()

    def sentida_score(sent):
        return sentida.sentida(sent, output ='total')     

    start = time.time()
    df_val['sentida'] = df_val.text.map(sentida_score).map(to_label_sentida)
    print_speed_performance(start, len(df_val))

    f1_report(df_val['polarity'], df_val['sentida'], 'Sentida', "twitter_sentiment(val)")


def bert_sent_benchmark():
    model = load_bert_tone_model()       

    start = time.time()
    preds = df_val.text.map(lambda x: model.predict(x))
    print_speed_performance(start, len(df_val))
    spellings_map = {'subjective': 'subjektivt', 'objective': 'objektivt', 'positive': 'positiv', 'negative': 'negativ', 'neutral': 'neutral'}
    df_val['bert_ana'] = preds.map(lambda x: spellings_map[x['analytic']])
    df_val['bert_pol'] = preds.map(lambda x: spellings_map[x['polarity']])

    f1_report(df_val['polarity'], df_val['bert_pol'], 'BERT_Tone (polarity)',  "twitter_sentiment(val)")
    f1_report(df_val['sub/obj'], df_val['bert_ana'], 'BERT_Tone (sub/obj)',  "twitter_sentiment(val)")

def spacy_benchmark():
    nlpS = load_spacy_model(textcat='sentiment', vectorError=True)

    # predict with spacy sentiment 
    def predict(x):
        doc = nlpS(x)
        return max(doc.cats.items(), key=operator.itemgetter(1))[0]
    
    start = time.time()
    df_val['spacy'] = df_val.text.map(lambda x: predict(x))
    print_speed_performance(start, len(df_val))

    f1_report(df_val['polarity'], df_val['spacy'], 'Spacy', "twitter_sentiment(val)")

if __name__ == '__main__':
    sentida_benchmark()
    afinn_benchmark()
    bert_sent_benchmark()
    spacy_benchmark()
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    