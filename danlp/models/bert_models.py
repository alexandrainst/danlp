import os
import re
from typing import Union, List
from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func
import torch
import warnings


class BertNer:
    """
    Bert NER model
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        from transformers import AutoModelForTokenClassification
        from transformers import AutoTokenizer

        # download the model or load the model path
        weights_path = download_model('bert.ner', cache_dir,
                                      process_func=_unzip_process_func,
                                      verbose=verbose)

        self.label_list = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG",
                           "I-ORG", "B-LOC", "I-LOC"]

        self.model = AutoModelForTokenClassification.from_pretrained(weights_path, num_labels = len(self.label_list))
        self.tokenizer = AutoTokenizer.from_pretrained(weights_path)

    def predict(self, text: Union[str, List[str]]):
        """
        Predict NER labels from raw text or tokenized text. If the text is
        a raw string this method will return the string tokenized with
        BERTs subword tokens.

        E.g. "varme vafler" will become ["varme", "va", "##fler"]

        :param text: Can either be a raw text or a list of tokens
        :return: The tokenized text and the predicted labels
        """

        if isinstance(text, str):
            # Bit of a hack to get the tokens with the special tokens
            tokens = self.tokenizer.tokenize(
                self.tokenizer.decode(self.tokenizer.encode(text)))
            inputs = self.tokenizer.encode(text, return_tensors="pt")
            if len(inputs[0]) > 512:
                warnings.warn(
                    'The Bert ner model can maximum take 512 tokens as input. Split instead you text before calling predict. Eg by using sentence boundary detection')

            outputs = self.model(inputs)[0]
            predictions = torch.argmax(outputs, dim=2)
            predictions = [self.label_list[label] for label in
                           predictions[0].tolist()]

            return tokens[1:-1], predictions[1:-1]  # Remove special tokens

        if isinstance(text, list):
            # Tokenize each word into the subword tokenization that BERT
            # uses. E.g. this tokenized text:
            #     tokens: ['Varme', 'vafler', 'og', 'friske', 'jordbær']
            # will get the following subword tokens and token_mask
            #     subwords: ['varme', 'va', '##fler', 'og', 'friske', 'jordbær']
            #     tokens_mask: [1, 1, 1, 0, 1, 1, 1, 1]
            tokens = []
            tokens_mask = [1]
            for word in text:
                word_tokens = self.tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                tokens_mask.extend([1]+[0]*(len(word_tokens)-1))
            tokens_mask.extend([1])

            inputs = self.tokenizer.encode(tokens, return_tensors="pt")
            if len(inputs[0]) > 512:
                warnings.warn(
                    'The Bert ner model can maximum take 512 tokens as input. Split instead you text before calling predict. Eg by using sentence boundary detection')

            assert inputs.shape[1] == len(tokens_mask)

            outputs = self.model(inputs)[0]

            predictions = torch.argmax(outputs, dim=2)

            # Mask the predictions so we only get the labels for the
            # pre-tokenized text
            predictions = [prediction for prediction, mask in zip(
                predictions[0].tolist(), tokens_mask) if mask]
            predictions = predictions[1:-1]  # Remove special tokens
            assert len(predictions) == len(text)

            # Map prediction ids to labels
            predictions = [self.label_list[label] for label in predictions]

            return text, predictions


class BertEmotion:
    """
    The class load both a BERT model to classify if emotion or not in the text,
    and a BERT model to regonizes eight emotions
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        from transformers import BertTokenizer, BertForSequenceClassification

        # download the model or load the model path
        path_emotion = download_model('bert.emotion', cache_dir,
                                      process_func=_unzip_process_func,
                                      verbose=verbose)
        path_emotion = os.path.join(path_emotion, 'bert.emotion')
        path_reject = download_model('bert.noemotion', cache_dir,
                                       process_func=_unzip_process_func,
                                       verbose=verbose)
        path_reject = os.path.join(path_reject,'bert.noemotion')


        # load the class names mapping
        self.catagories = {0: 'Glæde/Sindsro', 1: 'Tillid/Accept', 2: 'Forventning/Interrese',
                           3: 'Overasket/Målløs', 4: 'Vrede/Irritation', 5: 'Foragt/Modvilje', 6: 'Sorg/trist', 7: 'Frygt/Bekymret'}

        self.labels_no = {1: 'No emotion', 0: 'Emotional'}

        # load the models
        self.tokenizer_reject = BertTokenizer.from_pretrained(path_reject)
        self.model_reject = BertForSequenceClassification.from_pretrained(path_reject, num_labels=len(self.labels_no.keys()))
        
        self.tokenizer = BertTokenizer.from_pretrained(path_emotion)
        self.model = BertForSequenceClassification.from_pretrained(path_emotion, num_labels=len(self.catagories.keys()))
        
        # save embbeding dim, to later ensure the sequenze is no longer the embeddings 
        self.max_length = self.model.bert.embeddings.position_embeddings.num_embeddings
        self.max_length_reject = self.model_reject.bert.embeddings.position_embeddings.num_embeddings

    def _classes(self):
        return list(self.catagories.values()), list(self.labels_no.values())

    def _get_pred(self, tokenizer, model, max_lenght, sentence):
        input1 = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt',
                                       max_length=max_lenght, return_overflowing_tokens=True)
        if 'overflowing_tokens' in input1:
            warnings.warn(
                'Maximum length for sequence exceeded, truncation may result in unexpected results. Consider running the model on a shorter sequenze then {} tokens'.format(max_lenght))
        pred = model(input1['input_ids'],
                     token_type_ids=input1['token_type_ids'])[0]

        return pred

    def predict_if_emotion(self, sentence):

        pred = self._get_pred(
            self.tokenizer_reject, self.model_reject, self.max_length_reject, sentence)
        pred = pred.argmax().item()
        return self.labels_no[pred]

    def predict(self, sentence: str, no_emotion=False):

        def predict_emotion():
            pred = self._get_pred(
                self.tokenizer, self.model, self.max_length, sentence)
            pred = pred.argmax().item()
            return self.catagories[pred]

        if no_emotion:
            return predict_emotion()
        else:
            reject = self.predict_if_emotion(sentence)
            if reject == 'No emotion':
                return reject
            else:
                return predict_emotion()

    def predict_proba(self, sentence: str, emotions=True, no_emotion=True):
        proba = []

        # which emotion
        if emotions:
            pred = self._get_pred(
                self.tokenizer, self.model, self.max_length, sentence)
            proba.append(torch.nn.functional.softmax(
                pred[0], dim=0).detach().numpy())

        # emotion or no emotion
        if no_emotion:
            pred = self._get_pred(
                self.tokenizer_reject, self.model_reject, self.max_length_reject, sentence)
            proba.append(torch.nn.functional.softmax(
                pred[0], dim=0).detach().numpy())

        return proba


class BertTone:
    '''
    The class load both a BERT model to classify boteh the tone of [subjective or objective] and the tone og [positive, neutral , negativ]
    returns: [label_subjective, label_polarity]
    '''

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False, device=None, parallel: bool = False):
        """
        device (str | None): torch device the process should be run on. E.g. device = torch.device("cuda:0") to use the first GPU.
        parallel (bool): Should the process run in parallel (using torch.nn.DataParallel)
        """
        from transformers import BertTokenizer, BertForSequenceClassification

        # download the model or load the model path
        path_sub = download_model('bert.subjective', cache_dir, process_func=_unzip_process_func,verbose=verbose)
        path_sub = os.path.join(path_sub,'bert.sub.v0.0.1')
        path_pol = download_model('bert.polarity', cache_dir, process_func=_unzip_process_func,verbose=verbose)
        path_pol = os.path.join(path_pol,'bert.pol.v0.0.1')
        
        self.classes_pol= ['positive', 'neutral', 'negative']
        self.classes_sub= ['objective','subjective'] 

        self.tokenizer_sub = BertTokenizer.from_pretrained(path_sub)
        self.model_sub = BertForSequenceClassification.from_pretrained(path_sub, num_labels=len(self.classes_sub))
        self.tokenizer_pol = BertTokenizer.from_pretrained(path_pol)
        self.model_pol = BertForSequenceClassification.from_pretrained(path_pol, num_labels=len(self.classes_pol))
        
        # save embbeding dim, to later ensure the sequenze is no longer the embeddings 
        self.max_length_sub = self.model_sub.bert.embeddings.position_embeddings.num_embeddings
        self.max_length_pol = self.model_pol.bert.embeddings.position_embeddings.num_embeddings

        if device is not None:
            self.device = torch.device(device)
            self.model_sub.to(self.device)
            self.model_pol.to(self.device)
        self.parallel = parallel
        if self.parallel:
            self.model_sub = torch.nn.DataParallel(self.model_sub)
            self.model_pos = torch.nn.DataParallel(self.model_pos)

    def _clean(self, sentence):
        sentence = re.sub(r'http[^\s]+', '', sentence)
        sentence = re.sub(r'\n', '', sentence)
        sentence = re.sub(r'\t', '', sentence)
        sentence = re.sub(r'@', '', sentence)
        sentence = re.sub(r'#', '', sentence)
        return sentence

    def _classes(self):
        return self.classes_pol, self.classes_sub

    def _get_pred(self, tokenizer, model, max_length, sentence):
        if isinstance(sentence, str):  # to ensure common pipeline
            sentence = [sentence]
        input1 = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt',
                                       max_length=max_length, return_overflowing_tokens=True,
                                       truncation=True,
                                       padding=True)
        if 'overflowing_tokens' in input1:
            warnings.warn(
                'Maximum length for sequence exceeded, truncation may result in unexpected results. Consider running the model on a shorter sequenze then {} tokens'.format(max_length))
        if self.device:
            input1.to(self.device)  # move files to GPU if specified
        else:
            pred = model(**input1)[0]

        return pred

    def predict(self, sentence: str, polarity: bool = True, analytic: bool = True):

        sentence = self._clean(str(sentence))
        predDict = {'analytic': None, 'polarity': None}

        # predict subjective
        if analytic:
            #input1 = self.tokenizer_sub.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
            #pred = self.model_sub(input1['input_ids'], token_type_ids=input1['token_type_ids'])[0].argmax().item()
            pred = self._get_pred(
                self.tokenizer_sub, self.model_sub, self.max_length_sub, sentence)
            pred = pred.argmax(axis=1)
            res = [self.classes_sub[i] for i in pred]
            predDict['analytic'] = res if len(res) > 1 else res[0]

        # predict polarity
        if polarity:
            pred = self._get_pred(
                self.tokenizer_pol, self.model_pol, self.max_length_pol, sentence)
            pred = pred.argmax(axis=1)
            res = [self.classes_pol[i] for i in pred]
            #input1 = self.tokenizer_pol.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
            #pred = self.model_pol(input1['input_ids'], token_type_ids=input1['token_type_ids'])[0].argmax().item()
            predDict['polarity'] = res if len(res) > 1 else res[0]

        return predDict

    def predict_proba(self, sentence: str, polarity: bool = True, analytic: bool = True):
        proba = []
        sentence = self._clean(str(sentence))

        # predict polarity
        if polarity:
            pred = self._get_pred(
                self.tokenizer_pol, self.model_pol, self.max_length_pol, sentence)
            proba.append(torch.nn.functional.softmax(
                pred).cpu().detach().numpy())

        # predict subjective
        if analytic:
            pred = self._get_pred(
                self.tokenizer_sub, self.model_sub, self.max_length_sub, sentence)
            proba.append(torch.nn.functional.softmax(
                pred).cpu().detach().numpy())

        return proba


def load_bert_tone_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Wrapper function to ensure that all models in danlp are
    loaded in a similar way
    :param cache_dir:
    :param verbose:
    :return:
    """

    return BertTone(cache_dir, verbose)


def load_bert_emotion_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Wrapper function to ensure that all models in danlp are
    loaded in a similar way
    :param cache_dir:
    :param verbose:
    :return:
    """

    return BertEmotion(cache_dir, verbose)

def load_bert_ner_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Wrapper function to ensure that all models in danlp are
    loaded in a similar way
    :param cache_dir:
    :param verbose:
    :return:
    """
    return BertNer(cache_dir, verbose)
