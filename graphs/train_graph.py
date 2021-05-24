import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f128 = "u-128/ckpt-u-128-red-big-train.txt"
f256 = "u-256/ckpt-u-256-red-big-train.txt"
#f512 = "u-512/ckpt-u-512-final-train.txt"
#f1024 = "u-1024/ckpt-u-1024-final-train.txt"

data128 = pd.read_csv(f128)
data256 = pd.read_csv(f256)
# data512 = pd.read_csv(f512)
# data1024 = pd.read_csv(f1024)

col = "loss"
y128 = data128[col].to_numpy()[20:]
y256 = data256[col].to_numpy()[20:]
# y512 = data512[col].to_numpy()[20:]
# y1024 = data1024[col].to_numpy()[20:]
x = np.arange(20, 120, 1)

plt.plot(x, y128, '.-', label="u-128-red")
plt.plot(x, y256, '.-', label="u-256-red")
# plt.plot(x, y512, '.-', label="u-512")
# plt.plot(x, y1024, '.-', label="u-1024")
plt.legend(loc='upper right')
plt.show()