# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys, random
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

from lstm import lstm_unroll
from ocr_io import OCRIter

def Perplexity(label, pred):
    if random.random() < 0.01:
        print pred[0], label[0]
    label = label.T.reshape((-1,))
    hit = 0.0
    total = 0.0
    for i in range(pred.shape[0] / 4):
        ok = True
        for j in range(4):
            k = i * 4 + j
            if np.argmax(pred[k]) != int(label[k]):
                ok = False
                break
        if ok:
            hit += 1.0
        total += 1.0
    return hit / total

if __name__ == '__main__':
    batch_size = 50
    num_hidden = 200
    num_lstm_layer = 3

    num_epoch = 10
    learning_rate = 0.1
    momentum = 0.9

    contexts = [mx.context.gpu(0)]

    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len,
                              num_hidden=num_hidden,
                              num_label=10)

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = OCRIter(100000, batch_size, init_states)
    data_val = OCRIter(1000, batch_size, init_states)

    symbol = sym_gen(8)

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    model.fit(X=data_train, eval_data=data_val,
              eval_metric = mx.metric.np(Perplexity),
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),)

    model.save("ocr")
