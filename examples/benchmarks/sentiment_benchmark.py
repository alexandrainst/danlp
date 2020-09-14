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
               - pip install sentida
 
"""

from danlp.datasets import EuroparlSentiment1, LccSentiment
from danlp.models import load_bert_tone_model, load_spacy_model
from afinn import Afinn
import numpy as np
import tabulate
import os
import urllib
from pathlib import Path
import sys
import spacy
import operator

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
        return 'negative'
    else:
        return 'positive'
    
def to_label_sentida(score):
    # the treshold of 0.4 is fitted on a manuelt annotated twitter corpus for sentiment on 1327 exampels
    if score > 0.4:
        return 'positive'
    if score < -0.4:
        return 'negative'
    else:
        return 'neutral'


def afinn_benchmark(datasets):
    afinn = Afinn(language='da', emoticons=True)
    
    for dataset in datasets:
        if dataset == 'euparlsent':
            data = EuroparlSentiment1()
        if dataset == 'lccsent':
            data = LccSentiment()

        df = data.load_with_pandas()



        df['pred'] = df.text.map(afinn.score).map(to_label)
        df['valence'] = df['valence'].map(to_label)

        report(df['valence'], df['pred'], 'Afinn', dataset)
        
        
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



        df['pred'] = df.text.map(sentida_score).map(to_label_sentida)
        df['valence'] = df['valence'].map(to_label)

        report(df['valence'], df['pred'], 'Sentida', dataset)
        
def bert_sent_benchmark(datasets):
    model = load_bert_tone_model()
    
    for dataset in datasets:
        if dataset == 'euparlsent':
            data = EuroparlSentiment1()
        if dataset == 'lccsent':
            data = LccSentiment()

        df = data.load_with_pandas()


        df['valence'] = df['valence'].map(to_label)
        # predict with bert sentiment 
        df['pred'] = df.text.map(lambda x: model.predict(x, analytic=False)['polarity'])
        

        report(df['valence'], df['pred'], 'BERT_Tone (polarity)', dataset)

def spacy_sent_benchmark(datasets):
    
    nlpS = load_spacy_model(textcat='sentiment', vectorError=True)
   
    for dataset in datasets:
        if dataset == 'euparlsent':
            data = EuroparlSentiment1()
        if dataset == 'lccsent':
            data = LccSentiment()

        df = data.load_with_pandas()
        
        df['valence'] = df['valence'].map(to_label)
        
        # predict with spacy sentiment 
        def predict(x):
            doc = nlpS(x)
            pred = max(doc.cats.items(), key=operator.itemgetter(1))[0]
            #mathc the labels 
            labels = {'positiv': 'positive', 'neutral': 'neutral', 'negativ': 'negative'}
            return labels[pred]
        
        df['pred'] = df.text.map(lambda x: predict(x))
        

        report(df['valence'], df['pred'], 'Spacy sentiment (polarity)', dataset)
        
if __name__ == '__main__':
    sentida_benchmark(['euparlsent','lccsent'])
    afinn_benchmark(['euparlsent','lccsent'])
    bert_sent_benchmark(['euparlsent','lccsent'])
    spacy_sent_benchmark(['euparlsent','lccsent'])
