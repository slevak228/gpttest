import random

from autograd import Value

# ---------- Конфиг / матрицы / веса ----------
def build_config():
    n_embd = 16
    return dict(n_layer=1, n_embd=n_embd, block_size=16, n_head=4,
                head_dim=n_embd//4, vocab_size=0, BOS=0)

def make_matrix():
    return lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

def build_state_dict(cfg, matrix):
    sd = {'wte': matrix(cfg['vocab_size'], cfg['n_embd']),
          'wpe': matrix(cfg['block_size'], cfg['n_embd']),
          'lm_head': matrix(cfg['vocab_size'], cfg['n_embd'])}
    for i in range(cfg['n_layer']):
        sd[f'layer{i}.attn_wq'] = matrix(cfg['n_embd'], cfg['n_embd'])
        sd[f'layer{i}.attn_wk'] = matrix(cfg['n_embd'], cfg['n_embd'])
        sd[f'layer{i}.attn_wv'] = matrix(cfg['n_embd'], cfg['n_embd'])
        sd[f'layer{i}.attn_wo'] = matrix(cfg['n_embd'], cfg['n_embd'])
        sd[f'layer{i}.mlp_fc1'] = matrix(4*cfg['n_embd'], cfg['n_embd'])
        sd[f'layer{i}.mlp_fc2'] = matrix(cfg['n_embd'], 4*cfg['n_embd'])
    params = [p for m in sd.values() for row in m for p in row]
    return sd, params
