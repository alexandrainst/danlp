from danlp.datasets import DKHate
from danlp.models import load_bert_offensive_model, load_bert_hatespeech_model, load_electra_offensive_model
import time, os
from utils import *

## Load the DKHate data
dkhate = DKHate()
df_test, _ = dkhate.load_with_pandas()

sentences = df_test["tweet"].tolist()
labels_true = df_test["subtask_a"].tolist()
num_sentences = len(sentences)


def benchmark_bert_offensive_mdl():
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


def benchmark_attack_mdl():

    import torch
    from transformers import AutoTokenizer
    #from ogtal_model import ElectraClassifier

    from transformers import ElectraModel
    import torch.nn.functional as F
    import torch.nn as nn

    import wget

    class ElectraClassifier(nn.Module):
        
        def __init__(self,pretrained_model_name,num_labels=2):
            super(ElectraClassifier, self).__init__()
            self.num_labels = num_labels
            self.electra = ElectraModel.from_pretrained(pretrained_model_name)
            self.dense = nn.Linear(self.electra.config.hidden_size, self.electra.config.hidden_size)
            self.dropout = nn.Dropout(self.electra.config.hidden_dropout_prob)
            self.out_proj = nn.Linear(self.electra.config.hidden_size, self.num_labels)

        def classifier(self,sequence_output):
            x = sequence_output[:, 0, :]
            x = self.dropout(x)
            x = F.gelu(self.dense(x))
            x = self.dropout(x)
            x = F.gelu(self.dense(x))
            x = self.dropout(x)
            x = F.gelu(self.dense(x))
            x = self.dropout(x)
            logits = self.out_proj(x)
            return logits

        def forward(self, input_ids=None,attention_mask=None):
            discriminator_hidden_states = self.electra(input_ids=input_ids,attention_mask=attention_mask)
            sequence_output = discriminator_hidden_states[0]
            logits = self.classifier(sequence_output)
            return logits


    def make_prediction(text, tokzer, mdl):
        tokenized_text = tokzer(
            text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        input_ids = tokenized_text['input_ids']
        attention_masks = tokenized_text['attention_mask']
        logits = mdl(input_ids,attention_masks)
        
        _,preds = torch.max(logits, dim=1)
        return(int(preds))

    # load model
    model_checkpoint = 'Maltehb/-l-ctra-danish-electra-small-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = ElectraClassifier(model_checkpoint,2)
    mdir = 'examples/benchmarks'
    model_path = os.path.join(mdir, 'pytorch_model.bin')
    if not os.path.exists(model_path):
        url = 'https://github.com/ogtal/A-ttack/blob/main/pytorch_model.bin'
        print("Cannot find the model", model_path, "\nDownload the model at", url, 'and place it in directory', mdir)
        exit()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    labels = {0:'NOT', 1:'OFF'}

    start = time.time()

    preds = []
    for i, sentence in enumerate(sentences):
        pred = make_prediction(sentence, tokenizer, model)
        preds.append(labels[pred])

    print('OG TAL:')
    print_speed_performance(start, num_sentences)
    
    assert len(preds) == len(sentences)

    print(f1_report(labels_true, preds, "OgTal", "DR Data"))   


def benchmark_bert_hatespeech_mdl():
    bert_model = load_bert_hatespeech_model()

    start = time.time()

    preds = []
    for sentence in sentences:
        pred = bert_model.predict(sentence, offensive=True, hatespeech=False)
        preds.append(pred['offensive'])
    print('BERT Hatespeech:')
    print_speed_performance(start, num_sentences)
    
    assert len(preds) == num_sentences

    print(f1_report(labels_true, preds, "BERT", "DKHate"))


def benchmark_electra_offensive_mdl():
    electra_model = load_electra_offensive_model()

    start = time.time()

    preds = []
    for sentence in sentences:
        pred = electra_model.predict(sentence)
        preds.append(pred)
    print('Electra Offensive:')
    print_speed_performance(start, num_sentences)
    
    assert len(preds) == num_sentences

    print(f1_report(labels_true, preds, "electra", "DKHate"))

if __name__ == '__main__':
    benchmark_bert_offensive_mdl()
    benchmark_attack_mdl()
    benchmark_bert_hatespeech_mdl()
    benchmark_electra_offensive_mdl()

