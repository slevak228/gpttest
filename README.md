# gpttest

microgpt, разложенный на модули. Пайплайн — `pipeline(data)`.

## Модули

| Файл | Содержимое |
|---|---|
| `autograd.py` | класс `Value` |
| `primitives.py` | `linear`, `softmax`, `rmsnorm` |
| `tokenizer.py` | класс `Tokenizer` |
| `embedding.py` | класс `Embedding` |
| `gpt.py` | класс `GPT` |
| `config.py` | `build_config`, `make_matrix`, `build_state_dict` |
| `optimizer.py` | `build_optimizer`, `adam_update` |
| `training.py` | `train_loop`, `test_loop` |
| `pipeline.py` | `pipeline(data)` — только вызовы |
| `main.py` | точка входа |

## Запуск

```bash
python main.py
```

Python 3.10+. Если `input.txt` нет, скачается `names.txt` из `karpathy/makemore`.

## Проверки

```bash
python -m py_compile autograd.py primitives.py tokenizer.py embedding.py gpt.py config.py optimizer.py training.py pipeline.py main.py
python -c "import main, pipeline, training"
```

## CI

GitHub Actions запускает обе проверки при каждом push.
