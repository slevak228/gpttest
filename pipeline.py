from config import build_config, make_matrix, build_state_dict
from embedding import Embedding
from gpt import GPT
from optimizer import build_optimizer
from tokenizer import Tokenizer
from training import train_loop, test_loop

# ============================================================
# ПАЙПЛАЙН — ТОЛЬКО ВЫЗОВЫ
# ============================================================
def pipeline(data):
    cfg = build_config()
    matrix = make_matrix()

    # 1) ОБ ВЫБОРКА -> ТОКЕН -> СЛОВАРЬ ТОКЕНОВ
    tokenizer = Tokenizer().fit(data)
    cfg['vocab_size'], cfg['BOS'] = tokenizer.vocab_size, tokenizer.BOS

    # 2) СЛОВАРЬ ТОКЕНОВ -> ЭМБЕДДИНГ -> МАТРИЦА ЭМБЕДДИНГА
    embedding = Embedding(cfg['vocab_size'], cfg['block_size'], cfg['n_embd'], matrix)

    # 3) РАЗМЕРЫ/СТРУКТУРА -> СТРУКТУРА GPT -> ПРЕДСКАЗАНИЕ
    state_dict, params = build_state_dict(cfg, matrix)
    gpt = GPT(cfg, state_dict, cfg['n_head'], cfg['head_dim'])

    # 4) ОБУЧАЕМ -> ТЕСТ
    opt = build_optimizer(params)
    train_loop(gpt, embedding, tokenizer, data, cfg, params, opt)
    samples = test_loop(gpt, embedding, tokenizer, cfg)
    return samples
