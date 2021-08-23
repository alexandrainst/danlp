from danlp.datasets import DKHate
from danlp.models import load_bert_offensive_model
import time
from .utils import *

## Load the DKHate data
dkhate = DKHate()
df_test, _ = dkhate.load_with_pandas()

sentences = df_test["tweet"].tolist()
labels_true = df_test["subtask_a"].tolist()
num_sentences = len(sentences)


def benchmark_bert_mdl():
    bert_model = load_bert_offensive_model()

    start = time.time()

    preds = []
    for i, sentence in enumerate(sentences):
        pred = bert_model.predict(sentence)
        preds.append(pred)
    print('BERT:')
    print_speed_performance(start, num_sentences)
    
    assert len(preds) == num_sentences

    print(f1_report(labels_true, preds, "BERT", "DKHate"))    



if __name__ == '__main__':
    benchmark_bert_mdl()

