from danlp.datasets import DaNED
from danlp.models import load_xlmr_ned_model
from .utils import f1_report, print_speed_performance
import time

daned = DaNED()
_, _, test = daned.load_with_pandas()

sentences = test['sentence'].to_list()
qids = test['qid'].to_list()
kgs = [daned.get_kg_context_from_qid(qid)[0] for qid in qids]
gold_tags = [str(l) for l in test['class'].to_list()]

num_sentences = len(sentences)


def benchmark_xlmr_mdl():
    xlmr = load_xlmr_ned_model()

    start = time.time()

    predictions = []
    for sent,kg in zip(sentences, kgs):
        pred = xlmr.predict(sent, kg)
        predictions.append(pred)
    print('XLMR:')
    print_speed_performance(start, num_sentences)
    
    assert len(predictions) == num_sentences

    print(f1_report(gold_tags, predictions, 'XLM-R', 'DaNED'))


if __name__ == '__main__':
    benchmark_xlmr_mdl()