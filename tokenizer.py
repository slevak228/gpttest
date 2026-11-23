# ---------- 1) Токенизатор ----------
class Tokenizer:
    def __init__(self): self.uchars, self.BOS, self.vocab_size = [], 0, 0
    def fit(self, docs):
        self.uchars = sorted(set(''.join(docs)))
        self.BOS = len(self.uchars)
        self.vocab_size = len(self.uchars) + 1
        return self
    def encode(self, text):
        return [self.BOS] + [self.uchars.index(c) for c in text] + [self.BOS]
    def decode(self, ids):
        return ''.join(self.uchars[i] for i in ids if i != self.BOS)
