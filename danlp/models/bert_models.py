import os, re
from typing import Union, List
from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func
import torch
import warnings

class BertNer:
    """
    BERT NER model

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
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

    def predict(self, text: Union[str, List[str]], IOBformat=True):
        """
        Predict NER labels from raw text or tokenized text. If the text is
        a raw string this method will return the string tokenized with
        BERTs subword tokens.

        :param text: can either be a raw text or a list of tokens
        :param IOBformat: can either be TRUE or FALSE, but can only be False if text input is a list of tokens. Specifify if output should be in IOB format or a dictionary 
        :return: the tokenized text and the predicted labels in IOB format, or a dictionary with the tags and position

        :Example:

            "`varme vafler`" becomes ["varme", "va", "##fler"]
        """
        
        if isinstance(text, str):
            if IOBformat==False:
                warnings.warn('To choose a different output format with IOBformat=False, the text parameter should be a list of tokens like eg. ["Han", "hedder", "Ole"]')
            # Bit of a hack to get the tokens with the special tokens
            tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(text)))
            inputs = self.tokenizer.encode(text, return_tensors="pt")
            if len(inputs[0])>512:
                warnings.warn('The Bert ner model can maximum take 512 tokens as input. Split instead you text before calling predict. Eg by using sentence boundary detection')

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

            inputs = self.tokenizer.encode(text, return_tensors="pt", is_split_into_words=True)
            if len(inputs[0])>512:
                warnings.warn('The Bert ner model can maximum take 512 tokens as input. Split instead you text before calling predict. Eg by using sentence boundary detection')
            
            assert inputs.shape[1] == len(tokens_mask)

            outputs = self.model(inputs)[0]

            predictions = torch.argmax(outputs, dim=2)

            # Mask the predictions so we only get the labels for the
            # pre-tokenized text
            predictions = [prediction for prediction, mask in zip(predictions[0].tolist(), tokens_mask) if mask]
            predictions = predictions[1:-1]  # Remove special tokens
            assert len(predictions) == len(text)

            # Map prediction ids to labels
            predictions = [self.label_list[label] for label in predictions]
            
            if IOBformat==True:
                return text, predictions
                              
            else:
                text_combine = ""
                entities = []
                entity = None
                i = 0
                while i < len(predictions):
                    label = predictions[i]
                    if entity != None:
                        if label[0] == "I":
                            entity["text"] += ' {}'.format(text[i])
                        else:
                            entity["end_pos"] = len(text_combine) - 1
                            entities.append(entity)
                            entity = None
                    if label[0] == "B":
                        entity = {
                            "type": label.split("-")[1],
                            "text": '{}'.format(text[i]),
                            "start_pos": len(text_combine),
                            "end_pos": None
                        }
                    text_combine += '{} '.format(text[i])
                    i+=1
                text_combine = text_combine.rstrip()
                if entity != None:
                    entity["end_pos"] = len(text_combine)
                    entities.append(entity)
                return {"text": text_combine, "entities": entities}              
                
                              
                              

class BertEmotion:
    """
    BERT Emotion model.

    For classifying whether there is emotion in the text,
    and recognizing amongst eight emotions.

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        from transformers import BertTokenizer, BertForSequenceClassification
        
        # download the model or load the model path
        path_emotion = download_model('bert.emotion', cache_dir,
                                       process_func=_unzip_process_func,
                                       verbose=verbose)
        path_emotion = os.path.join(path_emotion,'bert.emotion')
        path_reject = download_model('bert.noemotion', cache_dir,
                                       process_func=_unzip_process_func,
                                       verbose=verbose)
        path_reject = os.path.join(path_reject,'bert.noemotion')

        # load the class names mapping
        self.catagories = { 0: 'Glæde/Sindsro',1: 'Tillid/Accept', 2: 'Forventning/Interrese',
              3: 'Overasket/Målløs',4: 'Vrede/Irritation', 5: 'Foragt/Modvilje', 6: 'Sorg/trist',7: 'Frygt/Bekymret'}

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
                                                max_length=max_lenght, truncation=True, return_overflowing_tokens=True)
        if 'overflowing_tokens' in input1 and input1['overflowing_tokens'].shape[1]>0:
            warnings.warn('Maximum length for sequence exceeded, truncation may result in unexpected results. Consider running the model on a shorter sequenze then {} tokens'.format(max_lenght))
        pred = model(input1['input_ids'], token_type_ids=input1['token_type_ids'])[0]

        return pred
    
    def predict_if_emotion(self, sentence):
        """
        Predicts whether there is emotion in the text.

        :param str sentence: raw sentence
        :return: 0 if no emotion else 1
        :rtype: int
        """
        
        pred=self._get_pred(self.tokenizer_reject, self.model_reject, self.max_length_reject, sentence)
        pred = pred.argmax().item()
        return self.labels_no[pred]
    
    def predict(self, sentence: str, no_emotion=False):
        """
        Predicts emotion among:

            * 0: `Glæde/Sindsro`
            * 1: `Tillid/Accept`
            * 2: `Forventning/Interrese`
            * 3: `Overasket/Målløs`
            * 4: `Vrede/Irritation`
            * 5: `Foragt/Modvilje`
            * 6: `Sorg/trist`
            * 7: `Frygt/Bekymret`

        :param str sentence: raw text
        :param bool no_emotion: whether there is emotion or not in the text
        :return: index of the emotion
        :rtype: int
        """
        
        def predict_emotion():
            pred=self._get_pred(self.tokenizer, self.model, self.max_length, sentence)
            pred = pred.argmax().item() 
            return self.catagories[pred]
        
        if no_emotion:
            return predict_emotion()
        else:
            reject=self.predict_if_emotion(sentence)
            if reject=='No emotion':
                return reject
            else:
                return predict_emotion()
            
    def predict_proba(self, sentence: str, emotions=True, no_emotion=True):
        """
        Predicts the probabilities of emotions.

        :param str sentence: raw text
        :param bool emotions: whether to return the probability of the emotion
        :param bool no_emotion: whether to return the probability of the sentence being emotional 
        :return: a list of probabilities
        :rtype: List
        """
        proba=[]
            
        # which emotion   
        if emotions:
            pred=self._get_pred(self.tokenizer, self.model, self.max_length, sentence)
            proba.append(torch.nn.functional.softmax(pred[0], dim=0).detach().numpy())
        
                
        # emotion or no emotion
        if no_emotion:
            pred=self._get_pred(self.tokenizer_reject, self.model_reject, self.max_length_reject, sentence)
            proba.append(torch.nn.functional.softmax(pred[0], dim=0).detach().numpy())

        return proba 

class BertTone:
    '''
    BERT Tone model. 

    For classifying both the tone [subjective, objective] 
    and the polarity [positive, neutral, negativ] of sentences.

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    '''

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
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

        
    def _clean(self, sentence):
        sentence=re.sub(r'http[^\s]+', '', sentence)
        sentence = re.sub(r'\n', '',sentence)
        sentence = re.sub(r'\t', '',sentence)
        sentence = re.sub(r'@', '',sentence)
        sentence = re.sub(r'#', '',sentence)
        return sentence

    def _classes(self):
        return self.classes_pol, self.classes_sub
    
    def _get_pred(self, tokenizer, model, max_lenght, sentence):
        input1 = tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt',
                                                max_length=max_lenght, truncation=True, return_overflowing_tokens=True)
        if 'overflowing_tokens' in input1 and input1['overflowing_tokens'].shape[1]>0:
            warnings.warn('Maximum length for sequence exceeded, truncation may result in unexpected results. Consider running the model on a shorter sequenze then {} tokens'.format(max_lenght))
        pred = model(input1['input_ids'], token_type_ids=input1['token_type_ids'])[0]

        return pred
    
    def predict(self, sentence: str, polarity: bool = True, analytic: bool = True):
        """
        Predict the polarity [positive, neutral, negativ] and/or the tone [subjective, objective] of the sentence. 

        :param str sentence: raw text
        :param bool polarity: returns the polarity if `True`
        :param bool analytic: returns the tone if `True`
        :return: a dictionary for polarity and tone results
        :rtype: Dict
        """
        
        sentence = self._clean(str(sentence))
        predDict = {'analytic': None, 'polarity': None }
        
        # predict subjective
        if analytic:
            #input1 = self.tokenizer_sub.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
            #pred = self.model_sub(input1['input_ids'], token_type_ids=input1['token_type_ids'])[0].argmax().item()
            pred=self._get_pred(self.tokenizer_sub, self.model_sub, self.max_length_sub, sentence)
            pred = pred.argmax().item()
            predDict['analytic']=self.classes_sub[pred]
        
        # predict polarity
        if polarity:
            pred=self._get_pred(self.tokenizer_pol, self.model_pol, self.max_length_pol, sentence)
            pred = pred.argmax().item()
            #input1 = self.tokenizer_pol.encode_plus(sentence, add_special_tokens=True, return_tensors='pt')
            #pred = self.model_pol(input1['input_ids'], token_type_ids=input1['token_type_ids'])[0].argmax().item() 
            predDict['polarity']=self.classes_pol[pred]
        
        return predDict
    
    def predict_proba(self, sentence: str, polarity: bool = True, analytic: bool = True):
        proba=[]
        sentence = self._clean(str(sentence))
        
        # predict polarity
        if polarity:
            pred=self._get_pred(self.tokenizer_pol, self.model_pol, self.max_length_pol, sentence)
            proba.append(torch.nn.functional.softmax(pred[0], dim=0).detach().numpy())
        
        # predict subjective
        if analytic:  
            pred=self._get_pred(self.tokenizer_sub, self.model_sub, self.max_length_sub, sentence)
            proba.append(torch.nn.functional.softmax(pred[0], dim=0).detach().numpy())

        return proba    

    
class BertBase:
    '''
    BERT language model used for embedding of tokens or sentence.
    The Model is trained by BotXO: https://github.com/botxo/nordic_bert
    The Bert model is transformed into pytorch version
    
    Credit for code example: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
    
    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    '''
    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        from transformers import BertTokenizer, BertModel
        import torch
        # download model
        self.path_model= download_model('bert.botxo.pytorch', cache_dir, process_func=_unzip_process_func,verbose=verbose)
        # Load pre-trained model tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.path_model)
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained(self.path_model,
                                          output_hidden_states = True, # Whether the model returns all hidden-states.
                                          )

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()

    def embed_text(self, text):
        """
        Calculate the embeddings for each token in a sentence ant the embedding for the sentence based on a BERT language model.
        The embedding for a token is chosen to be the concatenated last four layers, and the sentence embeddings to be the mean of the second to last layer of all tokens in the sentence
        The BERT tokenizer splits in subword for UNK word. The tokenized sentence is therefore returned as well. The embeddings for the special tokens are not returned.
       

        :param str sentence: raw text
        :return: three lists: token_embeddings (dim: tokens x 3072), sentence_embedding (1x738), tokenized_text
        :rtype: list, list, list
        """

        marked_text = "[CLS] " + text + " [SEP]"
        # Tokenize sentence with the BERT tokenizer
        tokenized_text = self.tokenizer.tokenize(marked_text)


        # Map the token strings to their vocabulary indeces
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)


        # Mark each of the tokens as belonging to sentence "1"
        segments_ids = [1] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

        token_embeddings = torch.stack(hidden_states, dim=0)
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # Swap dimensions 0 and 1. to tokens x layers x embedding
        token_embeddings = token_embeddings.permute(1,0,2)

        # choose to concatenate last four layers, dim 4x 768 = 3072
        token_vecs_cat= [torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0) for token in token_embeddings]
        # drop the CLS and the SEP tokens and embedding
        token_vecs_cat=token_vecs_cat[1:-1]
        tokenized_text =tokenized_text[1:-1]

        # chose to summarize the last four layers
        #token_vecs_sum=[torch.sum(token[-4:], dim=0) for token in token_embeddings]

        # sentence embedding
        # Calculate the average of all token vectors for the second last layers
        sentence_embedding = torch.mean(hidden_states[-2][0], dim=0)

        return token_vecs_cat, sentence_embedding, tokenized_text
    
class BertNextSent:
    '''
    BERT language model is trained for next sentence predictions.
    The Model is trained by BotXO: https://github.com/botxo/nordic_bert
    The Bert model is transformed into pytorch version
    
    Credit for code example: https://stackoverflow.com/questions/55111360/using-bert-for-next-sentence-prediction
    
    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    '''
    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        
        from transformers import BertForNextSentencePrediction, BertTokenizer
        # download model
        self.path_model= download_model('bert.botxo.pytorch', cache_dir, process_func=_unzip_process_func,verbose=verbose)
        # Load pre-trained model tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.path_model)
        # Load pre-trained model (weights)
        self.model = BertForNextSentencePrediction.from_pretrained(self.path_model,
                                          output_hidden_states = True, # Whether the model returns all hidden-states.
                                          )
        self.max_length = self.model.bert.embeddings.position_embeddings.num_embeddings

    def predict_if_next_sent(self, sent_A: str, sent_B: str):
        """
        Calculate the probability that sentence B follows sentence A.
        
        Credit for code example: https://stackoverflow.com/questions/55111360/using-bert-for-next-sentence-prediction
        
        :param str sent_A: sentence A
        :param str sent_B: sentence B
        :return: the probability of sentence B following sentence A 
        :rtype: float
        """
        from torch.nn.functional import softmax
        # encoded as "one" input to the model by using 'sent_B' as the 'text_pair'
        encoded = self.tokenizer.encode_plus(sent_A, text_pair=sent_B, return_tensors='pt',
                                        max_length=self.max_length, truncation=True, return_overflowing_tokens=True)
        if 'overflowing_tokens' in encoded and encoded['overflowing_tokens'].shape[1]>0:
             warnings.warn('Maximum length for sequence (sent_A + sent_B) exceeded, truncation may result in unexpected results. Consider running the model on a shorter sequence than {} tokens'.format(self.max_length))

        # a model's output is a tuple, we only need the output tensor containing
        # the relationships which is the first item in the tuple
        seq_relationship_logits = self.model(**encoded)[0]

        # we still need softmax to convert the logits into probabilities
        # index 0: sequence B is a continuation of sequence A
        # index 1: sequence B is a random sequence
        probs = softmax(seq_relationship_logits, dim=1)
        
        # return the pobability of sentence B following sentence A
        return round(float(probs[0][0]),4)



class BertOffensive():
    """
    BERT offensive language identification model.

    For predicting whether a text is offensive or not.

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        from transformers import BertTokenizer, BertForSequenceClassification
        # download the model or load the model path
        model_path = download_model('bert.offensive', cache_dir,
                                      process_func=_unzip_process_func,
                                      verbose=verbose)
        
        self.classes= ['NOT', 'OFF']
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(self.classes))
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.max_length = self.model.bert.embeddings.position_embeddings.num_embeddings

    def _classes(self):
        return self.classes
    
    def _get_pred(self, sentence):
        input1 = self.tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt',
                                                max_length=self.max_length, truncation=True, return_overflowing_tokens=True)
        if 'overflowing_tokens' in input1 and input1['overflowing_tokens'].shape[1]>0:
            warnings.warn('Maximum length for sequence exceeded, truncation may result in unexpected results. Consider running the model on a shorter sequence than {} tokens'.format(self.max_length))
        pred = self.model(input1['input_ids'], token_type_ids=input1['token_type_ids'])[0]

        return pred
    
    def predict(self, sentence: str):
        """
        Predict whether a text is offensive or not. 

        :param str sentence: raw text
        :return: a class -- `OFF` (offensive) or `NOT` (not offensive)
        :rtype: str
        """
        
        pred = self._get_pred(sentence)
        pred = pred.argmax().item()
        predclass = self.classes[pred]
        
        return predclass
    
    def predict_proba(self, sentence: str):
        """
        For a given sentence, 
        return its probabilities of belonging to each class, 
        i.e. `OFF` or `NOT`
        """

        pred=self._get_pred(sentence)
        proba = torch.nn.functional.softmax(pred[0], dim=0).detach().numpy()

        return proba



def load_bert_offensive_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Loads a BERT offensive language identification model.

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    :return: a BERT offensive language identification model
    """
    return BertOffensive(cache_dir, verbose)



def load_bert_nextsent_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Load BERT language model used for next sentence predictions.
    The Model is trained by BotXO: https://github.com/botxo/nordic_bert

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    :return: BERT NextSent model 
    """

    return BertNextSent(cache_dir, verbose)      
    
def load_bert_base_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Load BERT language model and use for embedding of tokens or sentence.
    The Model is trained by BotXO: https://github.com/botxo/nordic_bert

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    :return: BERT model 
    """

    return BertBase(cache_dir, verbose)  


def load_bert_tone_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Loads a BERT Tone model.

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    :return: a BERT Tone model
    """

    return BertTone(cache_dir, verbose)   
            
def load_bert_emotion_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Loads a BERT Emotion model.

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    :return: a BERT Emotion model
    """

    return BertEmotion(cache_dir, verbose)

def load_bert_ner_model(cache_dir=DEFAULT_CACHE_DIR, verbose=False):
    """
    Loads a BERT NER model.

    :param str cache_dir: the directory for storing cached models
    :param bool verbose: `True` to increase verbosity
    :return: a BERT NER model
    """
    return BertNer(cache_dir, verbose)
