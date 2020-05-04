from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from danlp.download import DEFAULT_CACHE_DIR, download_model, \
    _unzip_process_func


class BertNer:
    """
    Bert NER model
    """

    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, verbose=False):
        # download the model or load the model path
        weights_path = download_model('bert.ner', cache_dir,
                                      process_func=_unzip_process_func,
                                      verbose=verbose)

        self.label_list = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG",
                           "I-ORG", "B-LOC", "I-LOC"]

        self.model = AutoModelForTokenClassification.from_pretrained(weights_path)
        self.tokenizer = AutoTokenizer.from_pretrained(weights_path)

    def predict(self, text):
        # Bit of a hack to get the tokens with the special tokens
        tokens = self.tokenizer.tokenize(
            self.tokenizer.decode(self.tokenizer.encode(text)))
        inputs = self.tokenizer.encode(text, return_tensors="pt")

        outputs = self.model(inputs)[0]
        predictions = torch.argmax(outputs, dim=2)

        return [self.label_list[label] for label in predictions[0].tolist()]
