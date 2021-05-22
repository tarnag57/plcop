import numpy as np
from numpy import log
import os
import scipy
from scipy.sparse import vstack
from scipy.stats import loguniform, uniform
from sklearn.base import BaseEstimator
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from xgboost import sklearn


class XGBEstimator(BaseEstimator):

    def __init__(self, eta=0.3, max_depth=9, l2=1.5):
        self.eta = eta
        self.max_depth = max_depth
        self.l2 = l2        # This is the lambda param

    def fit(self, x, y):

        d_train = xgb.DMatrix(x, label=y)

        par = {}
        par['objective'] = "reg:linear"  # 'reg:linear' / 'binary:logistic'
        par['eval_metric'] = ['mae', 'rmse']
        par['eta'] = self.eta  # learning rate
        par['max_depth'] = self.max_depth  # max tree depth
        par['n_job'] = 4
        par['lambda'] = self.l2  # L2 weight decay
        par['silent'] = 1
        par['scale_pos_weight'] = 1
        num_round = 400
        evallist = [(d_train, 'train')]

        self.bst = xgb.train(par, d_train, num_round, evallist,
                             early_stopping_rounds=50, verbose_eval=10)

        return self

    def predict(self, x):
        d_predict = xgb.DMatrix(x)
        return self.bst.predict(d_predict)


def load_data(files, n_features):

    # Parse SVMLight files
    xs = []
    ys = []
    for f in files:
        try:
            print("Loading svmlight")
            d = load_svmlight_file(
                f, n_features=n_features, zero_based=True)
            xs.append(d[0])
            ys.append(d[1])
        except Exception as e:
            print(e)
            print(f"Was looking at file {f}")
    x = vstack(xs)
    y = np.concatenate(ys)

    print("Input shape: ", x.shape)
    print("Output shape: ", y.shape)
    print("Avg output: ", np.mean(y))

    remove_duplicates(x, y)

    return x, y


def remove_duplicates(x, y):
    '''
    Remove duplicate datapoints from a dataset. We always keep the largest associated label.

    :param x: The input features
    :param y: The corresponding labels
    '''
    global xs, ys   # TODO Where the hell are these variables coming from???
    dict = {}
    for i in range(x.shape[0]):
        s = str(x[i])
        if s in dict:
            if y[i] > dict[s]:
                dict[s] = y[i]
        else:
            dict[s] = y[i]

    xs = []
    ys = []
    for i in range(x.shape[0]):
        s = str(x[i])
        if dict[s] == y[i]:
            xs.append(x[i])
            ys.append(y[i])

    xs = vstack(xs)
    ys = np.array(ys)

    print("Reduced shapes {} and {} to {} and {}".format(
        x.shape, y.shape, xs.shape, ys.shape))
    return xs, ys


def listdir_fullpath(d):
    '''
    Returns the full path for each file in a given directory.

    :param d: The directory for which the files are returned.
    '''
    return [os.path.join(d, f) for f in os.listdir(d)]


train_type = "value"       # "value" or "policy"
train_dir = f"results/plcop/train_{train_type}"
files = listdir_fullpath(train_dir)

n_dim = 128
value_n_features = 2 * n_dim + 9
policy_n_features = 3 * n_dim + 9
n_features = value_n_features if train_type == "value" else policy_n_features

x, y = load_data(files, n_features)

param_space = dict(
    eta=loguniform(0.1, 0.5),
    max_depth=[7, 9, 11],
    l2=loguniform(0.8, 3)
)

model = XGBEstimator()
clf = RandomizedSearchCV(model, param_space, n_iter=10,
                         scoring="neg_mean_squared_error")
search = clf.fit(x, y)
print(search.best_params_)
print(search.best_score_)
