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
    

    def build_vocab(self, texts, max_size=30000, min_freq=1):
            counter = Counter()
            for t in texts:
                counter.update(self._basic_tokenize(t))
            most = [tok for tok, freq in counter.most_common(max_size) if freq >= min_freq]
            specials = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
            token_list = specials + most
            self.token2id = {t:i for i,t in enumerate(token_list)}
            self.id2token = {i:t for t,i in self.token2id.items()}


    def encode(self, text, add_special=True, max_len=None):
        toks = self._basic_tokenize(text)
        if add_special:
            toks = [self.bos_token] + toks + [self.eos_token]
        ids = [self.token2id.get(t, self.token2id.get(self.unk_token)) for t in toks]
        if max_len:
            ids = ids[:max_len]
        return ids
    

    def _basic_tokenize(self, text):
        text = text.strip()
        tokens = re.findall(r"\w+|[^\s\w]", text, flags=re.UNICODE)
        return tokens
    

    def decode(self, ids, skip_special=True):
        toks = [self.id2token.get(i, self.unk_token) for i in ids]
        if skip_special:
            toks = [t for t in toks if t not in {self.pad_token, self.bos_token, self.eos_token}]
        return " ".join(toks)
    

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)