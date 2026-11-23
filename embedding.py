# ---------- 2) Эмбеддинг ----------
class Embedding:
    def __init__(self, vocab_size, block_size, n_embd, matrix):
        self.wte = matrix(vocab_size, n_embd)
        self.wpe = matrix(block_size, n_embd)
    def forward(self, token_id, pos_id):
        return [t + p for t, p in zip(self.wte[token_id], self.wpe[pos_id])]
    def params(self):
        return [p for m in (self.wte, self.wpe) for row in m for p in row]
