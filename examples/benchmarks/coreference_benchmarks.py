
import time
from utils import print_speed_performance

from danlp.datasets import Dacoref
from danlp.models import load_xlmr_coref_model

from allennlp.data.data_loaders import SimpleDataLoader

import os

# load the data
dacoref = Dacoref()

_, _, testset = dacoref.load_as_conllu(predefined_splits=True)

num_sentences = len(testset)
num_tokens = sum([len(s) for s in testset])

def benchmark_xlmr_mdl():

    from allennlp.data import DataLoader
    from allennlp.training.util import evaluate

    xlmr = load_xlmr_coref_model()
    
    instances = xlmr.dataset_reader.load_dataset(testset)
    data_loader = SimpleDataLoader(instances, 1)
    data_loader.index_with(xlmr.model.vocab)

    start = time.time()

    metrics = evaluate(
        xlmr.model,
        data_loader
    )

    print('**XLM-R model**')
    print_speed_performance(start, num_sentences, num_tokens)
    print('Precision : ', metrics['coref_precision'])
    print('Recall : ', metrics['coref_recall'])
    print('F1 : ', metrics['coref_f1'])
    print('Mention Recall : ', metrics['mention_recall'])


if __name__ == '__main__':
    benchmark_xlmr_mdl()