"""
Evaluation script for sentiment analyis
**The scripts downloads scrips from a GitHub, and it is last tested on 24-03-2020**

The script benchmark on the following dataset where scores are converted into a three class problem: positiv, neutral, negative:
    - Europarl_sentiment
    - Lcc_sentiment 

The script benchmark the following models where scores are converted into a three class problem:
    - Afinn:
           Requirements:
               - afinn
    - SentidaV2:
           The script downloadsfilles from sentida github and place them in cache folder
           Requirement:
               - Pandas
               - NumPy
               - NLTK
                       
 
"""

from danlp.datasets import EuroparlSentiment, LccSentiment
from afinn import Afinn
import numpy as np
import tabulate
import os
import urllib
from pathlib import Path
import sys

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


def afinn_benchmark(datasets):
    afinn = Afinn(language='da', emoticons=True)
    
    for dataset in datasets:
        if dataset == 'euparlsent':
            data = EuroparlSentiment()
        if dataset == 'lccsent':
            data = LccSentiment()

        df = data.load_with_pandas()



        df['pred'] = df.text.map(afinn.score).map(to_label)
        df['valence'] = df['valence'].map(to_label)

        report(df['valence'], df['pred'], 'Afinn', dataset)
        
        
def sentida_benchmark(datasets):
    "The scripts download from github from sentindaV2 and place it in cache folder"
    DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), '.danlp')
    print(os.getcwd())
    workdir = DEFAULT_CACHE_DIR +'/sentida'
    print(workdir) 
    if not os.path.isdir(workdir):
        os.mkdir(workdir)
        url = "https://raw.githubusercontent.com/esbenkc/emma/master/SentidaV2/"
        for file in ['SentidaV2.py','aarup.csv','intensifier.csv']:
            urllib.request.urlretrieve(url+file, workdir+'/'+file)
                     
       
    
    sys.path.insert(1, workdir)
    os.chdir(workdir+ '/')
    sys.stdout = open(os.devnull, 'w')
    from SentidaV2 import sentidaV2
    sys.stdout = sys.__stdout__
    
    def sentida_score(sent):
        return sentidaV2(sent, output ='total')
    
    for dataset in datasets:
        if dataset == 'euparlsent':
            data = EuroparlSentiment()
        if dataset == 'lccsent':
            data = LccSentiment()

        df = data.load_with_pandas()



        df['pred'] = df.text.map(sentida_score).map(to_label)
        df['valence'] = df['valence'].map(to_label)

        report(df['valence'], df['pred'], 'SentidaV2', dataset)


if __name__ == '__main__':
    sentida_benchmark(['euparlsent','lccsent'])
    afinn_benchmark(['euparlsent','lccsent'])
    
