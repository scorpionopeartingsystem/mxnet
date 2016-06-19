# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import cv2, random, math
from io import BytesIO
from captcha.image import ImageCaptcha

class Batch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class DataIter(mx.io.DataIter):
    def __init__(self, fname, batch_size):
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.data = []
        for line in file(fname):
            tks = line.strip().split('\t')
            if len(tks) != 4:
                continue
            self.data.append((int(tks[0]), int(tks[1]), float(tks[2])))
        self.provide_data = [('user', (batch_size, )), ('item', (batch_size, ))]
        self.provide_label = [('score', (self.batch_size, ))]
        
    def __iter__(self):
        for k in range(len(self.data) / self.batch_size):
            users = []
            items = []
            scores = []
            for i in range(self.batch_size):
                j = k * self.batch_size + i
                user, item, score = self.data[j]
                users.append(user)
                items.append(item)
                scores.append(score)
                
            data_all = [mx.nd.array(users), mx.nd.array(items)]
            label_all = [mx.nd.array(scores)]
            data_names = ['user', 'item']
            label_names = ['score']
            
            data_batch = Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

def get_net(max_user, max_item):
    hidden = 500
    user = mx.symbol.Variable('user')
    item = mx.symbol.Variable('item')
    score = mx.symbol.Variable('score')

    user = mx.symbol.Embedding(data = user, input_dim = max_user, output_dim = 1000)
    user = mx.symbol.Flatten(data = user)
    user = mx.symbol.FullyConnected(data = user, num_hidden = hidden)
    item = mx.symbol.Embedding(data = item, input_dim = max_item, output_dim = 1000)
    item = mx.symbol.FullyConnected(data = item, num_hidden = hidden)
    item = mx.symbol.Flatten(data = item)
    pred = user * item
    pred = mx.symbol.sum_axis(data = pred, axis = 1)
    pred = mx.symbol.Flatten(data = pred)
    pred = mx.symbol.LinearRegressionOutput(data = pred, label = score)
    return pred

def max_id(fname):
    mu = 0
    mi = 0
    for line in file(fname):
        tks = line.strip().split('\t')
        if len(tks) != 4:
            continue
        mu = max(mu, int(tks[0]))
        mi = max(mi, int(tks[1]))
    return mu + 1, mi + 1

def RMSE(label, pred):
    ret = 0.0
    n = 0.0
    pred = pred.flatten()
    for i in range(len(label)):
        ret += (label[i] - pred[i]) * (label[i] - pred[i])
        n += 1.0
    return math.sqrt(ret / n)

if __name__ == '__main__':
    max_user, max_item = max_id('./data/u.data')
    print max_user, max_item
    network = get_net(max_user, max_item)
    devs = [mx.gpu(i) for i in range(1)]
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 100,
                                 learning_rate = 0.002,
                                 wd = 0.0001,
                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                                 momentum = 0.9)
    
    batch_size = 8
    data_train = DataIter('./data/u.train', batch_size)
    data_test = DataIter('./data/u.test', batch_size)
    
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    
    model.fit(X = data_train, eval_data = data_test, 
              eval_metric = RMSE,
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),)
    
    model.save("svd")
