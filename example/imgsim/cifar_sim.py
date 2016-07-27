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
    ret_data = {}
    for fname in fnames:
        fo = open(fname, 'rb')
        ret = cPickle.load(fo)
        fo.close()
        data = ret['data']
        labels = ret['labels']
        print len(labels)
        for i in range(len(labels)):
            la = labels[i]
            if la not in ret_data:
                ret_data[la] = []
            ret_data[la].append(data[i] / 255.0)
    print ret_data.keys()
    return ret_data

class DataIter(mx.io.DataIter):
    def __init__(self, data, batch_size):
        super(DataIter, self).__init__()
        self.data = data
        self.batch_size = batch_size
        self.provide_data = [('data1', (batch_size, 3, 32, 32)), \
                             ('data2', (batch_size, 3, 32, 32))]
        self.provide_label = [('label', (self.batch_size,))]
        
    def __iter__(self):
        print 'begin'
        keys = list(self.data.keys())
        size = sum([len(x) for _, x in self.data.items()]) / self.batch_size
        print size
        for i in range(size):
            data1_batch = []
            data2_batch = []
            label_batch = []
            for b in range(self.batch_size):
                j = keys[random.randint(0, len(keys) - 1)]
                k = keys[random.randint(0, len(keys) - 1)]
                if random.random() < 0.45:
                    k = j
                dj = self.data[j]
                dk = self.data[k]
                d1 = dj[random.randint(0, len(dj) - 1)]
                d2 = dk[random.randint(0, len(dk) - 1)]
                d1 = d1.reshape((3, 32, 32))
                d2 = d2.reshape((3, 32, 32))
                data1_batch.append(d1)
                data2_batch.append(d2)
                if k == j:
                    label_batch.append(1)
                else:
                    label_batch.append(0)
                    
            data_all = [mx.nd.array(data1_batch), mx.nd.array(data2_batch)]
            label_all = [mx.nd.array(label_batch)]
            data_names = ['data1', 'data2']
            label_names = ['label']
            
            data_batch = Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

def get_conv(data, conv_weight, conv_bias, fc_weight, fc_bias):
    cdata = data
    for i in range(3):
        cdata = mx.sym.Convolution(data=cdata, kernel=(3,3), num_filter=32,
                                   weight = conv_weight[i], bias = conv_bias[i],
                                   name = 'conv' + str(i))
        cdata = mx.sym.Pooling(data=cdata, pool_type="avg", kernel=(2,2), stride=(1, 1))
        cdata = mx.sym.Activation(data=cdata, act_type="relu")

    cdata = mx.sym.Flatten(data = cdata)
    cdata = mx.sym.FullyConnected(data = cdata, num_hidden = 1024,
                                  weight = fc_weight, bias = fc_bias, name='fc')
    return cdata


def get_net(batch_size):
    data1 = mx.sym.Variable('data1')
    data2 = mx.sym.Variable('data2')
    label = mx.sym.Variable('label')

    conv_weight = []
    conv_bias = []
    for i in range(3):
        conv_weight.append(mx.sym.Variable('conv' + str(i) + '_weight'))
        conv_bias.append(mx.sym.Variable('conv' + str(i) + '_bias'))
    fc_weight = mx.sym.Variable('fc_weight')
    fc_bias = mx.sym.Variable('fc_bias')
    fc1 = get_conv(data1, conv_weight, conv_bias, fc_weight, fc_bias)
    fc2 = get_conv(data2, conv_weight, conv_bias, fc_weight, fc_bias)

    pred = fc1 * fc2
    pred = mx.sym.FullyConnected(data = pred, num_hidden = 2)
    #label = mx.sym.Reshape(data = label, target_shape = (batch_size))
    #return mx.sym.LogisticRegressionOutput(data = pred, label = label)
    return mx.sym.SoftmaxOutput(data = pred, label = label)

class Auc(mx.metric.EvalMetric):
    def __init__(self):
        super(Auc, self).__init__('auc')

    def update(self, labels, preds):
        label = labels[0].asnumpy()
        pred = preds[0].asnumpy()
        if random.random() < 0.01:
            print pred[:10]
        return
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
    batch_size = 128
    network = get_net(batch_size)
    devs = [mx.gpu(i) for i in range(1)]
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 100,
                                 learning_rate = 0.01,
                                 wd = 0.00001,
                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                                 momentum = 0.0)
    dtrain = load_data(['./data/cifar-10-batches-py/data_batch_' + str(i) for i in range(1, 6)])
    dtest = load_data(['./data/cifar-10-batches-py/test_batch'])
    data_train = DataIter(dtrain, batch_size)
    data_test = DataIter(dtest, batch_size)
    
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    
    metric = Auc()
    model.fit(X = data_train, eval_data = data_test,
              #eval_metric = metric, 
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),)
    
    model.save("cifar-sim")
