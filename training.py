import random

from optimizer import adam_update
from primitives import softmax

# ---------- Обучение / тест ----------
def train_loop(gpt, emb, tok, docs, cfg, params, opt, num_steps=1000):
    for step in range(num_steps):
        tokens = tok.encode(docs[step % len(docs)])
        n = min(cfg['block_size'], len(tokens)-1)
        keys = [[] for _ in range(cfg['n_layer'])]
        vals = [[] for _ in range(cfg['n_layer'])]
        losses = []
        for pos in range(n):
            logits = gpt.forward(tokens[pos], pos, keys, vals, emb)
            probs = softmax(logits)
            losses.append(-probs[tokens[pos+1]].log())
        loss = (1/n) * sum(losses)
        loss.backward()
        adam_update(params, opt, step, num_steps)
        print(f"step {step+1:4d}/{num_steps} | loss {loss.data:.4f}", end='\r')

def test_loop(gpt, emb, tok, cfg, n_samples=20, temperature=0.5):
    out = []
    for _ in range(n_samples):
        keys = [[] for _ in range(cfg['n_layer'])]
        vals = [[] for _ in range(cfg['n_layer'])]
        tid, s = tok.BOS, []
        for pos in range(cfg['block_size']):
            logits = gpt.forward(tid, pos, keys, vals, emb)
            probs = softmax([l/temperature for l in logits])
            tid = random.choices(range(tok.vocab_size), weights=[p.data for p in probs])[0]
            if tid == tok.BOS: break
            s.append(tok.uchars[tid])
        out.append(''.join(s))
    return out
