# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys, random
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from operator import itemgetter

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

def load_data(fnames):
    import cPickle
    ret_data = []
    for fname in fnames:
        fo = open(fname, 'rb')
        ret = cPickle.load(fo)
        fo.close()
        data = ret['data'] / 255.0
        labels = ret['labels']
        ret_data += [(labels[i], data[i]) for i in range(len(data))]
    return ret_data

class DataIter(mx.io.DataIter):
    def __init__(self, data, batch_size):
        super(DataIter, self).__init__()
        self.data = data
        self.batch_size = batch_size
        self.provide_data = [('data', (batch_size, 3, 32, 32))]
        self.provide_label = [('softmax_label', (self.batch_size,))]
        
    def __iter__(self):
        count = len(self.data) / self.batch_size
        for i in range(count):
            data_batch = []
            label_batch = []
            for b in range(self.batch_size):
                l, d = self.data[random.randint(0, len(self.data) - 1)]
                d = d.reshape((3, 32, 32))
                data_batch.append(d)
                label_batch.append(l)
                    
            data_all = [mx.nd.array(data_batch)]
            label_all = [mx.nd.array(label_batch)]
            data_names = ['data']
            label_names = ['softmax_label']
            
            data_batch = Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

def get_conv(data):
    cdata = data
    ks = [5, 3, 3, 3]
    for i in range(len(ks)):
        cdata = mx.sym.Convolution(data=cdata, kernel=(ks[i],ks[i]), num_filter=32)
        if i == 0:
            cdata = mx.sym.Pooling(data=cdata, pool_type="max", kernel=(2,2), stride=(1, 1))
        else:
            cdata = mx.sym.Pooling(data=cdata, pool_type="avg", kernel=(2,2), stride=(1, 1))
        cdata = mx.sym.Activation(data=cdata, act_type="relu")

    cdata = mx.sym.Flatten(data = cdata)
    cdata = mx.sym.FullyConnected(data = cdata, num_hidden = 1024)
    cdata = mx.sym.Activation(data = cdata, act_type = "relu")
    return cdata


def get_net(batch_size):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    
    pred = get_conv(data)
    pred = mx.sym.FullyConnected(data = pred, num_hidden = 10)
    return mx.sym.SoftmaxOutput(data = pred, label = label)

class Auc(mx.metric.EvalMetric):
    def __init__(self):
        super(Auc, self).__init__('auc')

    def update(self, labels, preds):
        label = labels[0].asnumpy()
        pred = preds[0].asnumpy()
        tmp = [(label[i], pred[i] + 0.000001 * random.random()) for i in range(len(label))]
        tmp = sorted(tmp, key = itemgetter(1), reverse = True)
        if random.random() < 0.001:
            print tmp
        m = 0.0
        n = 0.0
        z = 0.0
        k = 0
        for a, b in tmp:
            if a > 0.5:
                m += 1.0
                z += len(tmp) - k
            else:
                n += 1.0
            k += 1
        z -= m * (m + 1.0) / 2.0
        if m == 0.0 or n == 0.0:
            return 0.5
        z /= m
        z /= n
        self.sum_metric += z
        self.num_inst += 1

if __name__ == '__main__':
    batch_size = 256
    import symbol_resnet
    network = symbol_resnet.get_symbol(10)
    devs = [mx.gpu(i) for i in range(1)]
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 100,
                                 learning_rate = 0.01,
                                 wd = 0.00001,
                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                                 momentum = 0.9)
    dtrain = load_data(['./data/cifar-10-batches-py/data_batch_' + str(i) for i in range(1, 6)])
    dtest = load_data(['./data/cifar-10-batches-py/test_batch'])
    data_train = DataIter(dtrain, batch_size)
    data_test = DataIter(dtest, batch_size)
    
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    
    metric = Auc()
    model.fit(X = data_train, eval_data = data_test,
#              eval_metric = metric, 
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),)
    
    model.save("cifar-cls")
