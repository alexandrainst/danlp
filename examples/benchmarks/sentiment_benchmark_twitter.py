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
import numpy as np
import tabulate
import os
import urllib
from pathlib import Path
import sys
import operator

## Load teh witter data
twitSent = TwitterSent()
df_val, _ = twitSent.load_with_pandas()


def f1_class(k, true, pred):
    tp = np.sum(np.logical_and(pred == k, true == k))

    fp = np.sum(np.logical_and(pred == k, true != k))
    fn = np.sum(np.logical_and(pred != k, true == k))
    if tp == 0:
        return 0
    recall = tp / (tp + fp)
    precision = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def report(true, pred, modelname, dataname):
    data_b = []
    data_a = []
    headers_b = ["{} // {} ".format(modelname, dataname), 'Class', 'Precission', 'Recall', 'F1']
    headers_a = ['Accuracy', 'Avg-f1', 'Weighted-f1', '', '']
    aligns_b = ['left', 'left', 'center', 'center', 'center']
    
    acc = np.sum(true == pred) / len(true)

    n = len(np.unique(true))
    avg = 0
    wei = 0
    for c in np.unique(true):
        precision, recall, f1 = f1_class(c, pred, true)
        avg += f1 / n
        wei += f1 * (np.sum(true == c) / len(true))

        data_b.append(['', c, precision, recall, f1])
    data_b.append(['', '', '', '', ''])
    data_b.append(headers_a)
    data_b.append([acc, avg, wei, '', ''])
    print()
    print( tabulate.tabulate(data_b, headers=headers_b, colalign=aligns_b), '\n')
    
    
def to_label(score):
    if score == 0:
        return 'neutral'
    if score < 0:
        return 'negativ'
    else:
        return 'positiv'
    
def to_label_sentida(score):
    # the treshold of 0.4 is fitted on a manuelt annotated twitter corpus for sentiment on 1327 exampels
    if score > 0.4:
        return 'positiv'
    if score < -0.4:
        return 'negativ'
    else:
        return 'neutral'
    
    
def afinn_benchmark():
    from afinn import Afinn
    afinn = Afinn(language='da', emoticons=True)
    df_val['afinn'] = df_val.text.map(afinn.score).map(to_label)

    report(df_val['polarity'], df_val['afinn'], 'Afinn', "twitter_sentiment(val)")
        
        
def sentida_benchmark():

    from sentida import Sentida
    sentida =  Sentida()
    
    def sentida_score(sent):
        return sentida.sentida(sent, output ='total')     
        
    df_val['sentida'] = df_val.text.map(sentida_score).map(to_label_sentida)

    report(df_val['polarity'], df_val['sentida'], 'Sentida', "twitter_sentiment(val)")
        
        
def bert_sent_benchmark():
    model = load_bert_tone_model()       
        
    preds = df_val.text.map(lambda x: model.predict(x))
    spellings_map = {'subjective': 'subjektivt', 'objective': 'objektivt', 'positive': 'positiv', 'negative': 'negativ', 'neutral': 'neutral'}
    df_val['bert_ana'] = preds.map(lambda x: spellings_map[x['analytic']])
    df_val['bert_pol'] = preds.map(lambda x: spellings_map[x['polarity']])
    
    report(df_val['polarity'], df_val['bert_pol'], 'BERT_Tone (polarity)',  "twitter_sentiment(val)")    
    report(df_val['sub/obj'], df_val['bert_ana'], 'BERT_Tone (sub/obj)',  "twitter_sentiment(val)")      
        
def spacy_benchmark():
    nlpS = load_spacy_model(textcat='sentiment', vectorError=True)
    
    # predict with spacy sentiment 
    def predict(x):
        doc = nlpS(x)
        return max(doc.cats.items(), key=operator.itemgetter(1))[0]
    
    df_val['spacy'] = df_val.text.map(lambda x: predict(x))

    report(df_val['polarity'], df_val['spacy'], 'Spacy', "twitter_sentiment(val)")        
        
if __name__ == '__main__':
    sentida_benchmark()
    afinn_benchmark()
    bert_sent_benchmark()
    spacy_benchmark()
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    