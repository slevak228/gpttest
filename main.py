import os
import random

from pipeline import pipeline

random.seed(42)

# ---------- Точка входа ----------
if __name__ == "__main__":
    if not os.path.exists('input.txt'):
        import urllib.request
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt',
            'input.txt')
    docs = [l.strip() for l in open('input.txt') if l.strip()]
    random.shuffle(docs)
    print(f"num docs: {len(docs)}")
    samples = pipeline(docs)
    print("\n--- inference ---")
    for i, s in enumerate(samples, 1):
        print(f"sample {i:2d}: {s}")
