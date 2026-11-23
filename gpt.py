from primitives import linear, rmsnorm, softmax

# ---------- 3) GPT ----------
class GPT:
    def __init__(self, cfg, state_dict, n_head, head_dim):
        self.cfg, self.sd = cfg, state_dict
        self.n_head, self.head_dim = n_head, head_dim
        self.n_layer = cfg['n_layer']
    def forward(self, token_id, pos_id, keys, values, emb):
        x = rmsnorm(emb.forward(token_id, pos_id))
        for li in range(self.n_layer):
            x_res = x; x = rmsnorm(x)
            q = linear(x, self.sd[f'layer{li}.attn_wq'])
            k = linear(x, self.sd[f'layer{li}.attn_wk'])
            v = linear(x, self.sd[f'layer{li}.attn_wv'])
            keys[li].append(k); values[li].append(v)
            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs:hs+self.head_dim]
                k_h = [ki[hs:hs+self.head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs+self.head_dim] for vi in values[li]]
                logits = [sum(q_h[j]*k_h[t][j] for j in range(self.head_dim)) / self.head_dim**0.5
                          for t in range(len(k_h))]
                w = softmax(logits)
                x_attn.extend([sum(w[t]*v_h[t][j] for t in range(len(v_h))) for j in range(self.head_dim)])
            x = linear(x_attn, self.sd[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_res)]
            x_res = x; x = rmsnorm(x)
            x = linear(x, self.sd[f'layer{li}.mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, self.sd[f'layer{li}.mlp_fc2'])
            x = [a + b for a, b in zip(x, x_res)]
        return linear(x, self.sd['lm_head'])
