# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import random, math
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

def read_text8():
    with open("./data/text8") as f:
        tks = f.read().split(' ')
        freq = {}
        for tk in tks:
            if tk not in freq:
                freq[tk] = 1
            else:
                freq[tk] += 1
        vocab = {}
        n = 0
        for k, v in sorted(freq.items(), key = itemgetter(1), reverse = True):
            vocab[k] = n
            n += 1
        return vocab, freq, [vocab[x] for x in tks]

class DataIter(mx.io.DataIter):
    def __init__(self, fname, batch_size, win_size):
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.vocab, self.freq, self.data = read_text8()
        self.vocab_size = len(self.vocab)
        self.provide_data = [('context', (batch_size, win_size * 2)), \
                             ('target', (batch_size, 1))]
        self.provide_label = [('label', (self.batch_size, ))]
        
    def __iter__(self):
        batch_context = []
        batch_target = []
        batch_label = []
        for k in range(len(self.data) - 2 * win_size - 1):
            sentence = self.data[k : (k + 2 * win_size - 1)]
            context = self.data[k : (k + win_size)] \
                      + self.data[(k+win_size + 1): (k + 2 * win_size + 1)]
            target = sentence[win_size]
            batch_context.append(context)
            batch_target.append([target])
            batch_label.append(1)
            
            for j in range(5):
                batch_context.append(context)
                batch_target.append([self.data[random.randint(0, len(self.data) - 1)]])
                batch_label.append(0)

            if len(batch_label) == batch_size:
                data_all = [mx.nd.array(batch_context),
                            mx.nd.array(batch_target)]
                label_all = [mx.nd.array(batch_label)]
                data_names = ['context', 'target']
                label_names = ['label']
                batch_label = []
                batch_context = []
                batch_target = []
                
                data_batch = Batch(data_names, data_all, label_names, label_all)
                yield data_batch

    def reset(self):
        pass

def get_net(win_size, vocab_size):
    context = mx.symbol.Variable('context')
    target = mx.symbol.Variable('target')
    label = mx.symbol.Variable('label')
    
    context_words = mx.symbol.Embedding(data = context,
                                        input_dim = vocab_size,
                                        output_dim = 200)
    context_words = mx.symbol.SliceChannel(data = context_words,
                                           num_outputs = win_size * 2,
                                           squeeze_axis = 1)
    context_vec = []
    for i in range(win_size * 2):
        context_vec.append(context_words[i])
    context_vec = mx.symbol.ElementWiseSum(*context_vec)
    context_vec = context_vec
    context_vec = mx.symbol.Flatten(data = context_vec)
    context_vec = mx.symbol.FullyConnected(data = context_vec, num_hidden = 200)

    target = mx.symbol.Embedding(data = target, input_dim = vocab_size,
                                 output_dim = 200)
    target = mx.symbol.Flatten(data = target)
    target = mx.symbol.FullyConnected(data = target, num_hidden = 200)
    pred = context_vec * target
    pred = mx.symbol.sum(data = pred, axis = 1, keepdims = True)
    pred = mx.symbol.Flatten(data = pred)
    pred = mx.symbol.LogisticRegressionOutput(data = pred, label = label)
    return pred

def AUC(label, pred):
    pred = pred.reshape(-1)
    lp = [(label[i], pred[i]) for i in range(len(label))]
    lp = sorted(lp, key = itemgetter(1), reverse = True)
    a = 0.0
    m = 0.0
    n = 0.0
    for i in range(len(lp)):
        if lp[i][0] > 0.5:
            m += 1
            a += len(lp) - i
        else:
            n += 1
    return (a - m * (m+1)) / (m * n)

if __name__ == '__main__':
    batch_size = 6 * 256
    win_size = 5
    data = DataIter('./data/text8', batch_size, win_size)
    print data.vocab_size

    network = get_net(win_size = win_size, vocab_size = data.vocab_size)
    devs = [mx.gpu(i) for i in range(4)]
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 100,
                                 learning_rate = 0.01,
                                 wd = 0.0001,
                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                                 momentum = 0.9)
    
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    
    model.fit(X = data, eval_metric = AUC,
              kvstore = 'local_allreduce_device',
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),)
    
    model.save("svd")
