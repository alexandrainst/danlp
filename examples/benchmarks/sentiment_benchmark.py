from danlp.datasets import EuroParlSent, LccSent
from afinn import Afinn
import numpy as np
import tabulate

def f1_class(k, true, pred):

    tp = np.sum(np.logical_and(pred== k, true == k))
    
    fp = np.sum(np.logical_and(pred == k, true != k))
    fn = np.sum(np.logical_and(pred != k, true == k))
    if tp== 0:
        return 0
    recall = tp / (tp + fp)
    precision  = tp / (tp + fn)
    f1=2*(precision*recall)/(precision+recall)
    return precision, recall, f1

def report(true, pred, modelname):
    dataB = []
    dataA= []
    headersB = ['class', 'precission', 'recall', 'f1']
    headersA = ['model', 'accuracy', 'avg-f1', 'weighted-f1']
    aligns = ['left', 'center', 'center', 'center']
    
    acc = np.sum(true==pred)/len(true)
    
    N = len(np.unique(true))
    avg = 0
    wei=0
    for c in np.unique(true):
        precision, recall, f1= f1_class(c,pred,true) 
        avg += f1/ N
        wei += f1 *(np.sum(true==c)/len(true)) 
        
        dataB.append([c,precision, recall, f1])
    dataA.append([modelname, acc, avg, wei])
    
    print(tabulate.tabulate(dataA, headers=headersA, tablefmt='github', colalign=aligns))
    print()
    print(tabulate.tabulate(dataB, headers=headersB, tablefmt='github', colalign=aligns), '\n')
    
def to_label(score):
    if score == 0:
        return 'neutral'
    if score< 0:
        return 'negativ'
    else:
        return 'positiv'
    
    
def afinn_benchmark(dataset):
    
    if dataset=='euparlsent':
        print('Eutoparl sentiment evaluation set annotated by Finn Årup Nielsen \n')
        data = EuroParlSent()
    if dataset=='lccsent':
        print('Lcc sentiment evaluation set annotated by Finn Årup Nielsen \n')
        data = LccSent()      
    
              
    df = data.load_with_pandas()
    
    afinn = Afinn(language='da', emoticons=True)
    
    df['affin']=df.text.map(afinn.score).map(to_label)
    df['valence']=df['valence'].map(to_label)
    
    report(df['valence'], df['affin'], 'Afinn')
              
              
if __name__ == '__main__':
    afinn_benchmark('euparlsent')
    afinn_benchmark('lccsent')
             
              
              
              
    