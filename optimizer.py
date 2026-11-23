# ---------- Оптимизатор ----------
def build_optimizer(params):
    return dict(lr=0.01, beta1=0.85, beta2=0.99, eps=1e-8,
                m=[0.0]*len(params), v=[0.0]*len(params))

def adam_update(params, opt, step, num_steps):
    lr_t = opt['lr'] * (1 - step / num_steps)
    for i, p in enumerate(params):
        opt['m'][i] = opt['beta1']*opt['m'][i] + (1-opt['beta1'])*p.grad
        opt['v'][i] = opt['beta2']*opt['v'][i] + (1-opt['beta2'])*p.grad**2
        mh = opt['m'][i] / (1 - opt['beta1']**(step+1))
        vh = opt['v'][i] / (1 - opt['beta2']**(step+1))
        p.data -= lr_t * mh / (vh**0.5 + opt['eps'])
        p.grad = 0
