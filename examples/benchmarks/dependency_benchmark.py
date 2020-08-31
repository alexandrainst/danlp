
import time

from danlp.datasets import DDT
from danlp.models import load_spacy_model

from tabulate import tabulate


# load the data
ddt = DDT()

ccorpus_conll = ddt.load_as_conllu(predefined_splits=True)
deps_true = []
# the test set
sentences_tokens = []
for sent in ccorpus_conll[2]:
    sentences_tokens.append([token.form for token in sent._tokens])
    deps_true.append([(token.deprel.lower(),int(token.head)) for token in sent._tokens])
num_sentences = len(sentences_tokens)
num_tokens = sum([len(s) for s in sentences_tokens])


def print_dependency_scores(deps_true, deps_pred):

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
        la_score_per_label.append(correct_deps[l]["rel"]/correct_deps[l]["total"])
        uas_score_per_label.append(correct_deps[l]["head"]/correct_deps[l]["total"])
        las_score_per_label.append(correct_deps[l]["both"]/correct_deps[l]["total"])
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

    print("\n", tabulate(tab, headers=headers, colalign=["right", "decimal", "decimal", "decimal", "right"]), "\n")


def benchmark_spacy_mdl():

    def normalize_spacy_head(i, hd):
        return 0 if i == hd else hd+1

    nlp = load_spacy_model()
    parser = nlp.parser
    
    start = time.time()

    deps_pred = []
    for sent in sentences_tokens:
        doc = nlp.tokenizer.tokens_from_list(sent)
        doc = parser(doc)

        deprels = []
        depheads = []
        for i, tok in enumerate(doc):
            deprels.append(tok.dep_.lower())
            depheads.append(normalize_spacy_head(i, tok.head.i))
        deps_pred.append([(r,h) for r,h in zip(deprels, depheads)])

    print('**Spacy model**')
    print("Made predictions on {} sentences and {} tokens in {}s".format(
    num_sentences, num_tokens, time.time() - start))
    
    assert len(deps_pred)==num_sentences
    assert sum([len(s) for s in deps_pred])==num_tokens
    
    print_dependency_scores(deps_true, deps_pred)


if __name__ == '__main__':
    benchmark_spacy_mdl()