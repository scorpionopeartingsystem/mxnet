# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import cv2

def get_ocrnet():
    data = mx.symbol.Variable('data')
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=32)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5,5), num_filter=32)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3,3), num_filter=32)
    pool3 = mx.symbol.Pooling(data=conv3, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=pool3, act_type="relu")
    
    flatten = mx.symbol.Flatten(data = relu3)
    fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 512)
    fc2 = mx.symbol.FullyConnected(data = fc1, num_hidden = 40)
    return mx.symbol.LogisticRegressionOutput(fc2, name = 'softmax')


def CRPS(label, pred):
    ret = 0
    n = 0
    for i in range(pred.shape[0]):
        for j in range(4):
            start = j * 10
            end = j * 10 + 10
            if np.argmax(pred[i][start:end]) == np.argmax(label[i][start:end]):
                ret += 1.0
            n += 1.0
    return ret / n
    #for i in range(pred.shape[0]):
    #    for j in range(pred.shape[1] - 1):
    #        if pred[i, j] > pred[i, j + 1]:
    #            pred[i, j + 1] = pred[i, j]
    #return np.sum(np.square(label - pred)) / label.size    
    
network = get_ocrnet()
devs = [mx.gpu(0)]
model = mx.model.FeedForward(ctx = devs,
                             symbol = network,
                             num_epoch = 5,
                             learning_rate = 0.001,
                             wd = 0.00001,
                             initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                             momentum = 0.9)
data_train = mx.io.ImageRecordIter(
    batch_size=32,
    path_imgrec="data/train.bin",
    data_shape=(3,60,160),
    path_imglist="data/train.list",
    label_width=40
)


import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

model.fit(X = data_train, eval_metric = mx.metric.np(CRPS), batch_end_callback=mx.callback.Speedometer(32, 50),)
