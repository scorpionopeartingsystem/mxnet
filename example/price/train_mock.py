# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys, datetime, math, random
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from io import BytesIO

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

def interval(price):
    if price < 1:
        return 0
    elif price >= 1 and price < 2:
        return 1
    elif price >= 2 and price < 4:
        return 2
    elif price >= 4 and price < 8:
        return 3
    elif price >= 8 and price < 16:
        return 4
    elif price >= 16 and price < 32:
        return 5
    elif price >= 32 and price < 64:
        return 6
    else:
        return 7
    

class PriceIter(mx.io.DataIter):
    def __init__(self, fname, batch_size):
        super(PriceIter, self).__init__()
        self.batch_size = batch_size
        self.dis = []
        self.series = []
        self.price = []
        for line in file(fname):
            price, d, s = line.strip().split("\t")
            self.price.append(float(price))
            self.series.append(np.array([int(s)], dtype = np.int))
            self.dis.append(np.array([float(d) / 10.0]))
        self.provide_data = [('dis', (batch_size, 1)),
                             ('series', (batch_size, 1))]
        self.provide_label = [('price', (batch_size, )),
                              ('price_interval', (batch_size,))]

    def __iter__(self):
        count = len(self.price)
        for i in range(count / self.batch_size):
            bdis = []
            bseries = []
            blabel = []
            blabel_interval = []
            for j in range(self.batch_size):
                k = i * self.batch_size + j
                bdis.append(self.dis[k])
                bseries.append(self.series[k])
                blabel.append(self.price[k])
                blabel_interval.append(interval(self.price[k]))

            data_all = [mx.nd.array(bdis),
                        mx.nd.array(bseries)]
            label_all = [mx.nd.array(blabel), mx.nd.array(blabel_interval)]
            data_names = ['dis', 'series']
            label_names = ['price', 'price_interval']
            
            data_batch = Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

def get_net():
    dis = mx.symbol.Variable('dis')
    price = mx.symbol.Variable('price')
    price_interval = mx.symbol.Variable('price_interval')
    series = mx.symbol.Variable('series')

    dis = mx.symbol.Flatten(data = dis, name = "dis_flatten")
    series = mx.symbol.Embedding(data = series, input_dim = 200,
                                 output_dim = 100, name = "series_embed")
    series = mx.symbol.Flatten(series, name = "series_flatten")

    net = mx.symbol.Concat(*[dis, series], dim = 1, name = "concat")
    net = mx.symbol.FullyConnected(data = net, num_hidden = 100, name = "fc1")
    net = mx.symbol.Activation(data = net, act_type="relu")
    net = mx.symbol.FullyConnected(data = net, num_hidden = 100, name = "fc2")
    net = mx.symbol.Activation(data = net, act_type="relu")
    net = mx.symbol.FullyConnected(data = net, num_hidden = 1, name = "fc3")
    net = mx.symbol.Activation(data = net, act_type="relu")
    net = mx.symbol.LinearRegressionOutput(data = net, label = price, name = "lro")

    net2 = mx.symbol.Concat(*[dis, series], dim = 1, name = "concat")
    net2 = mx.symbol.FullyConnected(data = net2, num_hidden = 100, name = "fc21")
    net2 = mx.symbol.Activation(data = net2, act_type="relu")
    net2 = mx.symbol.FullyConnected(data = net2, num_hidden = 100, name = "fc22")
    net2 = mx.symbol.Activation(data = net2, act_type="relu")
    net2 = mx.symbol.FullyConnected(data = net2, num_hidden = 8, name = "fc23")
    net2 = mx.symbol.Activation(data = net2, act_type="relu")
    net2 = mx.symbol.SoftmaxOutput(data = net2, label = price_interval, name="sf")
    return mx.symbol.Group([net, net2])

def RMSE(label, pred):
    ret = 0.0
    n = 0.0
    if pred.shape[1] == 8:
        return None
    for k in range(pred.shape[0]):
        v1 = label[k]
        v2 = pred[k][0]
        ret += abs(v1 - v2) / v1
        n += 1.0
    return ret / n

if __name__ == '__main__':
    data_train = PriceIter("mock.train", 100)
    data_test = PriceIter("mock.test", 100)
    
    network = get_net()
    
    devs = [mx.gpu(i) for i in range(1)]
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 10,
                                 learning_rate = 0.0001,
                                 wd = 0.0001,
                                 lr_scheduler=mx.misc.FactorScheduler(2000,0.9),
                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                                 momentum = 0.9)

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    model.fit(X = data_train, eval_data = data_test, eval_metric = mx.metric.np(RMSE), batch_end_callback=mx.callback.Speedometer(32, 50),)
    model.save('mock')
