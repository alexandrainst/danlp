

import time

from tabulate import tabulate
import numpy as np
from seqeval.metrics import classification_report


def accuracy_report(tags_true, tags_pred, per_label=True):

    # flatening tags lists
    tags_true = [tag for sent in tags_true for tag in sent]
    tags_pred = [tag for sent in tags_pred for tag in sent]

    # list of all tags
    labels = sorted(list(set(tags_true)))

    headers = ["label", "accuracy", "support"]
    tab = []
    correct_tags = {l:[] for l in labels}
    # counting correct predictions per tag
    for label in labels:
        correct_tags[label] = [t == p for t, p in zip(tags_true, tags_pred) if t == label]
        acc = round(sum(correct_tags[label])/len(correct_tags[label])*100, 2) if len(correct_tags[label])>0 else 0
        tab.append([label, acc, len(correct_tags[label])])
    if per_label:
        tab.append(['', '', ''])
    total_examples = sum(len(correct_tags[l]) for l in correct_tags)

    micro_acc = round( sum(sum(correct_tags[l]) for l in correct_tags) / total_examples *100, 2)
    tab.append(["micro average", micro_acc, total_examples])

    macro_acc = round( sum([sum(correct_tags[l])/len(correct_tags[l]) for l in labels if len(correct_tags[l])>0]) / len(labels) *100, 2)
    tab.append(["macro average", macro_acc, total_examples])

    return tabulate(tab, headers=headers, colalign=["right", "decimal", "right"])


def dependency_report(deps_true, deps_pred):

    # list of all labels (= dependency relations)
    labels = sorted(list(set([r for s in deps_true for (r,_) in s]+[r for s in deps_pred for (r,_) in s])))

    # counting (per label) of the correctly predicted dependencies (relations, heads and relation+head)
    correct_deps = {l:{"total" : 0, "rel" : 0, "head": 0, "both": 0} for l in labels}
    for sent_true, sent_pred in zip(deps_true, deps_pred):
        for (rt, ht), (rp, hp) in zip(sent_true, sent_pred):
            correct_deps[rt]["total"] += 1 
            if rt == rp:
                correct_deps[rt]["rel"] += 1
            if ht == hp:
                correct_deps[rt]["head"] += 1
            if rt == rp and ht == hp:
                correct_deps[rt]["both"] += 1

    # LA = label accuracy
    # UAS = unlabelled attachment score
    # LAS = labelled attachment score
    headers = ["label", "LA", "UAS", "LAS", "total"]
    tab, la_score_per_label, uas_score_per_label, las_score_per_label = [], [], [], []
    for l in labels:
        correct_deps_total_not_zero = bool(correct_deps[l]["total"])
        la_score_per_label.append(correct_deps[l]["rel"]/correct_deps[l]["total"] if correct_deps_total_not_zero else 0)
        uas_score_per_label.append(correct_deps[l]["head"]/correct_deps[l]["total"] if correct_deps_total_not_zero else 0)
        las_score_per_label.append(correct_deps[l]["both"]/correct_deps[l]["total"] if correct_deps_total_not_zero else 0)
        tab.append([l, round(la_score_per_label[-1]*100,2), round(uas_score_per_label[-1]*100,2), round(las_score_per_label[-1]*100,2), correct_deps[l]["total"]])
    tab.append(['']*5)

    total_examples = sum(correct_deps[l]["total"] for l in correct_deps)

    micro_la = round( sum(correct_deps[l]["rel"] for l in correct_deps) / total_examples *100, 2)
    micro_uas = round( sum(correct_deps[l]["head"] for l in correct_deps) / total_examples *100, 2)
    micro_las = round( sum(correct_deps[l]["both"] for l in correct_deps) / total_examples *100, 2)
    tab.append(["micro average", micro_la, micro_uas, micro_las, total_examples])

    macro_la = round( sum(la_score_per_label) / len(la_score_per_label) *100, 2)
    macro_uas = round( sum(las_score_per_label) / len(uas_score_per_label) *100, 2)
    macro_las = round( sum(las_score_per_label) / len(uas_score_per_label) *100, 2)
    tab.append(["macro average", macro_la, macro_uas, macro_las, total_examples])

    return tabulate(tab, headers=headers, colalign=["right", "decimal", "decimal", "decimal", "right"])



def f1_class(k, true, pred):
    tp = np.sum(np.logical_and(pred == k, true == k))

    fp = np.sum(np.logical_and(pred == k, true != k))
    fn = np.sum(np.logical_and(pred != k, true == k))
    if tp == 0:
        return 0, 0, 0, 0, 0, 0
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    return tp, fp, fn, precision, recall, f1


def f1_report(true, pred, modelname="", dataname="", word_level=False, bio=False):

    if bio:
        return classification_report(true, pred, digits=4)

    if word_level:
        true = [tag for sent in true for tag in sent]
        pred = [tag for sent in pred for tag in sent]

    data_b = []
    data_a = []
    headers_b = ["{} // {} ".format(modelname, dataname), 'Class', 'Precision', 'Recall', 'F1', 'support']
    headers_a = ['Accuracy', 'Avg-f1', 'Weighted-f1', '', '']
    aligns_b = ['left', 'left', 'center', 'center', 'center']

    true = np.array(true)
    pred = np.array(pred)
    acc = np.sum(true == pred) / len(true)

    n = len(np.unique(true))
    avg = 0
    wei = 0
    for c in np.unique(true):
        _, _, _, precision, recall, f1 = f1_class(c, true, pred)
        avg += f1 / n
        wei += f1 * (np.sum(true == c) / len(true))

        data_b.append(['', c, round(precision, 4), round(recall, 4), round(f1, 4)])
    data_b.append(['', '', '', '', ''])
    data_b.append(headers_a)
    data_b.append([round(acc, 4), round(avg, 4), round(wei, 4), '', ''])
    print()
    print(tabulate(data_b, headers=headers_b, colalign=aligns_b), '\n')


def print_speed_performance(start, num_sentences, num_tokens=None):

    span = time.time() - start
    sent_per_sec = int(num_sentences/span)
    span = round(span, 5)
    if not num_tokens == None :
        speed  = "Made predictions on {} sentences and {} tokens in {}s (~{} sentences per second)"
        speed = speed.format(num_sentences, num_tokens, span, sent_per_sec)
    else: 
        speed  = "Made predictions on {} sentences in {}s (~{} sentences per second)"
        speed = speed.format(num_sentences, span, sent_per_sec)
    print(speed)


def sentiment_score_to_label(score):
    if score == 0:
        return 'neutral'
    if score < 0:
        return 'negativ'
    else:
        return 'positiv'

def sentiment_score_to_label_sentida(score):
    # the threshold of 0.4 is fitted on a manually annotated twitter corpus for sentiment on 1327 examples
    if score > 0.4:
        return 'positiv'
    if score < -0.4:
        return 'negativ'
    else:
        return 'neutral'
