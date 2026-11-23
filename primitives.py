# ---------- Примитивы ----------
def linear(x, w): return [sum(wi*xi for wi, xi in zip(wo, x)) for wo in w]
def softmax(logits):
    mx = max(v.data for v in logits)
    exps = [(v - mx).exp() for v in logits]
    tot = sum(exps)
    return [e / tot for e in exps]
def rmsnorm(x):
    ms = sum(xi*xi for xi in x) / len(x)
    s = (ms + 1e-5) ** -0.5
    return [xi*s for xi in x]
