import re
from collections import Counter
import json
import os


class SimpleTokenizer:
    def __init__(self, vocab=None, unk_token="<unk>", pad_token="<pad>", bos_token="<s>", eos_token="</s>"):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        if vocab:
            self.token2id = vocab
            self.id2token = {i:t for t,i in vocab.items()}
        else:
            self.token2id = {}
            self.id2token = {}


    def _basic_tokenize(self, text):
        text = text.strip()
        tokens = re.findall(r"\w+|[^\s\w]", text, flags=re.UNICODE)
        return tokens