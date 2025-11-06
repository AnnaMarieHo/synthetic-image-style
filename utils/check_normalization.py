import numpy as np
data = np.load("openfake-annotation/datasets/combined/cache/pure_style_embeddings.npz")
style = data["style"]          # 25-D normalized vectors
label = data["label"]

real = style[label == 1]
fake = style[label == 0]

print("Mean | Std (real):")
print(real.mean(axis=0))
print(real.std(axis=0))

print("Mean | Std (fake):")
print(fake.mean(axis=0))
print(fake.std(axis=0))
