# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys, random, os
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

class DataIter(mx.io.DataIter):
    def __init__(self, names, batch_size):
        super(DataIter, self).__init__()
        self.names = names
        self.cache = []
        random.shuffle(self.names)
        for i in range(100):
            self.cache.append(np.load(self.names[i]))
        print 'load data ok'
        self.batch_size = batch_size
        self.provide_data = [('data1', (batch_size, 3, 64, 64)), \
                             ('data2', (batch_size, 3, 64, 64))]
        self.provide_label = [('label', (self.batch_size,))]
        
    def generate_same(self, n):
        ck = random.randint(0, 99)
        data = self.cache[ck]
        ret = []
        while len(ret) < n:
            k1 = random.randint(0, len(data) - 1)
            k2 = random.randint(0, len(data) - 1)
            if k1 == k2:
                continue
            ret.append((data[k1], data[k2]))
        return ret

    def generate_diff(self, n):
        n1, n2 = random.sample(range(100), 2)
        d1 = self.cache[n1]
        d2 = self.cache[n2]
        ret = []
        while len(ret) < n:
            k1 = random.randint(0, len(d1) - 1)
            k2 = random.randint(0, len(d2) - 1)
            ret.append((d1[k1], d2[k2]))
        return ret

    def __iter__(self):
        print 'begin'
        for i in range(10000):
            same = self.generate_same(self.batch_size / 2)
            diff = self.generate_diff(self.batch_size / 2)
            data1_batch = [x[0] for x in same] + [x[0] for x in diff]
            data2_batch = [x[1] for x in same] + [x[1] for x in diff]
            label_batch = [1 for _ in range(len(same))] + [0 for _ in range(len(diff))]
            
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
    return mx.sym.SoftmaxOutput(data = pred, label = label)

class Auc(mx.metric.EvalMetric):
    def __init__(self):
        super(Auc, self).__init__('auc')

    def update(self, labels, preds):
        label = labels[0].asnumpy()
        pred = preds[0].asnumpy()
        print label
        print pred
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
    batch_size = 512
    network = get_net(batch_size)
    devs = [mx.gpu(i) for i in range(4)]
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 100,
                                 learning_rate = 0.01,
                                 wd = 0.00001,
                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                                 momentum = 0.0)
    names = []
    root = '/data/service_data/ILSVRC2012_train_tar'
    for fn in os.listdir(root):
        if fn.endswith('.npy'):
            names.append(root + '/' + fn)
    print len(names)
    data_train = DataIter(names, batch_size)
    
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    
    metric = Auc()
    model.fit(X = data_train, #eval_data = data_test,
#              eval_metric = metric, 
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),)
    
    model.save("cifar-sim")
