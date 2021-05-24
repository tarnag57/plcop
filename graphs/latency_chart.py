import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

categories = ['u-128', 'u-256']

total = np.array([1611, 2380])
preprocess = np.array([287, 289])
encoding = np.array([1217, 1762])
overhead = total - preprocess - encoding

ind = [x for x, _ in enumerate(categories)]

plt.bar(ind, overhead, label='overhead', bottom=preprocess+encoding)
plt.bar(ind, encoding, label='encoding', bottom=preprocess)
plt.bar(ind, preprocess, label='preprocess')

plt.xticks(ind, categories)
plt.ylabel("Time μs")

plt.legend(loc="upper left")
plt.show()