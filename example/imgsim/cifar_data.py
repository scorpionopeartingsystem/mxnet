import cPickle
import numpy as np

fnames = ['./data/cifar-10-batches-py/test_batch']

groups = {}
for name in fnames:
    fo = open(name, 'rb')
    ret = cPickle.load(fo)
    fo.close()
    data = ret['data']
    labels = ret['labels']
    for i in range(len(labels)):
        la = labels[i]
        if la not in groups:
            groups[la] = []
        groups[la].append(data[i].reshape((3, 32, 32)) / 255.0)

for k, imgs in groups.items():
    out = "./data/cifar/test" + str(k)
    np.save(out, np.array(imgs))

