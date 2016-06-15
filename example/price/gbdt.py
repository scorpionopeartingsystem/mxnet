from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_svmlight_file
import numpy as np

def load(fname):
    maxk = 0
    ret = []
    for line in file(fname):
        tks = line.strip().split('\t')
        price = float(tks[0])
        vals = {}
        for x in tks[1:]:
            k, v = x.split(':')
            k = int(k)
            v = float(v)
            vals[k] = v
            maxk = max(maxk, k)
        ret.append((price, vals))
    return maxk, ret

mk_train, train = load("train.tsv.svm")
mk_test, test = load("test.tsv.svm")

mk = max(mk_train, mk_test) + 1

X_train = np.zeros((len(train), mk))
y_train = np.zeros((len(train), ))

for i in range(len(train)):
    price, vals = train[i]
    y_train[i] = price
    for k, v in vals.items():
        X_train[i][k] = v

X_test = np.zeros((len(test), mk))
y_test = np.zeros((len(test), 1))
for i in range(len(test)):
    price, vals = test[i]
    y_test[i] = price
    for k, v in vals.items():
        X_test[i][k] = v


print X_train.shape, y_train.shape
print X_test.shape, y_test.shape

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, \
                                 max_depth=1, random_state=0).fit(X_train, y_train)
clf.score(X_test, y_test)
