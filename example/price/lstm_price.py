# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys, datetime, math, random
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np

from collections import namedtuple
import time
import math
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])

class Batch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

    @property
    def provide_data(self):
        print [(n, x.shape) for n, x in zip(self.data_names, self.data)]
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        print [(n, x.shape) for n, x in zip(self.label_names, self.label)]
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

def process_line(line):
    price, deal_price, create_time, car_id, minor_category_name, \
        license_date, license_month, road_haul, city_id, \
        pinpai, chexi, chexing, guobie, zhidaojiage, \
        niankuan, cheliangleixing, cheliangjibie, biansuxiangleixing, \
        cheshenxingshi = line.strip('\n').split('\t')
    deal_price = float(deal_price)
    price = float(price)
    ratio = min(price, deal_price) / max(price, deal_price)
    #price = 0.0
    #if deal_price > 0. and ratio > 0.8:
    #    price = deal_price
    date_delta = datetime.datetime.fromtimestamp(int(create_time)).date() - datetime.date(int(license_date),int(license_month),15)
    vehicle_age = date_delta.days
    pinpai = pinpai.replace(" ", "").replace(":", "")
    chexi = chexi.replace(" ", "").replace(":", "")
    chexi = pinpai + "_" + chexi
    chexing = chexing.replace(" ", "").replace(":", "")
    chexing = chexi + "_" + chexing
    vehicle_age /= 365.0
    road_haul = float(road_haul) / 10000.0
    zhidaojiage_tks = zhidaojiage.split("-")
    zhidaojiage_min = 0
    zhidaojiage_max = 0
    try:
        zhidaojiage_min = float(zhidaojiage_tks[0])
        zhidaojiage_max = float(zhidaojiage_tks[-1])
    except Exception, e:
        pass
    return price / 10000.0, vehicle_age, 'pp_' + pinpai, 'chexi_' + chexi, 'chexing_' + chexing, road_haul, city_id, \
        zhidaojiage_min, zhidaojiage_max, 'gb_' + guobie, 'nk_' + niankuan, 'cllx_' + cheliangleixing, 'cljb_' + cheliangjibie, 'bsxlx_' + biansuxiangleixing, 'csxs_' + cheshenxingshi

def insert_vocab(a, b):
    if b not in a:
        a[b] = len(a)

def build_vocab(fname):
    vchexing = {}
    vcity = {}
    for line in file(fname):
        price, vehicle_age, pinpai, chexi, chexing, road_haul, city_id, \
            zhidaojiage_min, zhidaojiage_max, guobie, niankuan, cheliangleixing, cheliangjibie, biansuxiangleixing, cheshenxingshi = process_line(line)
        insert_vocab(vchexing, pinpai)
        insert_vocab(vchexing, chexi)
        insert_vocab(vchexing, chexing)
        insert_vocab(vchexing, guobie)
        insert_vocab(vchexing, niankuan)
        insert_vocab(vchexing, cheliangleixing)
        insert_vocab(vchexing, cheliangjibie)
        insert_vocab(vchexing, biansuxiangleixing)
        insert_vocab(vchexing, cheshenxingshi)
        insert_vocab(vcity, city_id)
    return vchexing, vcity


def save_vocab(vocab, fname):
    sw = open(fname, "w")
    for k, v in vocab.items():
        sw.write(str(k) + "\t" + str(v) + "\n")
    sw.close()

class PriceIter(mx.io.DataIter):
    def __init__(self, fname, batch_size, vchexing, vcity, init_states):
        super(PriceIter, self).__init__()
        self.batch_size = batch_size
        self.default_bucket_key = 6
        self.city = [[] for i in range(self.default_bucket_key + 1)]
        self.chexing = [[] for i in range(self.default_bucket_key + 1)]
        self.cdata = [[] for i in range(self.default_bucket_key + 1)]
        self.price = [[] for i in range(self.default_bucket_key + 1)]
        self.bucket = [0 for i in range(self.default_bucket_key + 1)]
        self.chexing_vocab = vchexing
        self.city_vocab = vcity
        for line in file(fname):
            tks = line.split("||")
            self.bucket[len(tks)] += 1
            s_cdata = []
            s_chexing = []
            s_city = []
            s_price = []
            for tk in tks:
                price, vehicle_age, pinpai, chexi, chexing, road_haul, city_id, \
                    zhidaojiage_min, zhidaojiage_max, guobie, niankuan, cheliangleixing, cheliangjibie, biansuxiangleixing, cheshenxingshi = process_line(tk)
                if price <= 0.:
                    continue
                s_cdata += [vehicle_age / 10.0, float(road_haul)/ 10.0,
                                zhidaojiage_min / 10.0, zhidaojiage_max / 10.0]
                s_chexing += [self.chexing_vocab[pinpai], 
                                  self.chexing_vocab[chexi], 
                                  self.chexing_vocab[chexing],
                                  self.chexing_vocab[guobie],
                                  self.chexing_vocab[niankuan],
                                  self.chexing_vocab[cheliangleixing],
                                  self.chexing_vocab[cheliangjibie],
                                  self.chexing_vocab[biansuxiangleixing],
                                  self.chexing_vocab[cheshenxingshi]
                              ]
                s_city += [self.city_vocab[city_id]]
                s_price += [price]
            bucket = len(tks)
            self.cdata[bucket].append(s_cdata)
            self.chexing[bucket].append(s_chexing)
            self.city[bucket].append(s_city)
            self.price[bucket].append(s_price)
        
        self.plan = []
        for i, n in enumerate(self.bucket):
            s = n / batch_size
            for j in range(s):
                self.plan.append((i, j))
        
        random.shuffle(self.plan)
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [('cdata', (batch_size, 4 * self.default_bucket_key)),
                             ('chexing', (batch_size, 9 * self.default_bucket_key)),
                             ('city', (batch_size, 1 * self.default_bucket_key))] + init_states
        self.provide_label = [('price', (self.batch_size, self.default_bucket_key))]

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]
        for i, s in self.plan:
            bcdata = []
            bcity = []
            bchexing = []
            blabel = []
            for j in range(self.batch_size):
                k = s * self.batch_size + j
                bcdata.append(self.cdata[i][k])
                bcity.append(self.city[i][k])
                bchexing.append(self.chexing[i][k])
                blabel.append(self.price[i][k])
                
            data_all = [mx.nd.array(bcdata), mx.nd.array(bchexing), mx.nd.array(bcity)] + self.init_state_arrays
            label_all = [mx.nd.array(blabel)]
            data_names = ['cdata', 'chexing', 'city'] + init_state_names
            label_names = ['price']
            
            data_batch = Batch(data_names, data_all, label_names, label_all, i)
            yield data_batch

    def reset(self):
        pass

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx):
    i2h = mx.symbol.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.symbol.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.symbol.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.symbol.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.symbol.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.symbol.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.symbol.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.symbol.Activation(next_c, act_type="relu")
    return LSTMState(c=next_c, h=next_h)

def get_net(chexing_size, chexing_embed_size,
            city_size, city_embed_size, seq_len, num_lstm_layer):
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    cdata = mx.symbol.Variable('cdata')
    price = mx.symbol.Variable('price')
    chexing = mx.symbol.Variable('chexing')
    city = mx.symbol.Variable('city')

    cdata = mx.symbol.Flatten(data = cdata)
    cdata = mx.symbol.SliceChannel(data = cdata, num_outputs = seq_len, squeeze_axis = 1)

    chexing_embed = mx.symbol.Embedding(data = chexing, 
                                        input_dim = chexing_size,
                                        output_dim = chexing_embed_size,
                                        name="chexing_embed")
    chexing_flatten = mx.symbol.Flatten(chexing_embed)
    chexing_flatten = mx.symbol.SliceChannel(data = chexing_flatten, num_outputs = seq_len, squeeze_axis = 1)
    city_embed = mx.symbol.Embedding(data = city, input_dim = city_size, 
                                     output_dim = city_embed_size,
                                     name="city_embed")
    city_flatten = mx.symbol.Flatten(city_embed)
    city_flatten = mx.symbol.SliceChannel(data = city_flatten, num_outputs = seq_len, squeeze_axis = 1)
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = mx.symbol.Concat(*[cdata[seqidx], chexing_flatten[seqidx], city_flatten[seqidx]], dim = 1)

        # stack LSTM
        for i in range(num_lstm_layer):
            next_state = lstm(500, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i)
            hidden = next_state.h
            last_states[i] = next_state
        hidden_all.append(hidden)

    hidden_concat = mx.symbol.Concat(*hidden_all, dim=0)
    pred = mx.symbol.FullyConnected(data=hidden_concat, num_hidden=1,
                                 weight=cls_weight, bias=cls_bias, name='pred')

    label = mx.sym.transpose(data=price)
    label = mx.sym.Reshape(data=label, target_shape=(0,))
    sm = mx.sym.LinearRegressionOutput(data=pred, label=label)
    return sm

def RMSE(label, pred):
    ret = 0.0
    n = 0.0
    label = label.transpose().flatten()
    pred = pred.flatten()
    for k in range(pred.shape[0]):
        v1 = label[k]
        v2 = pred[k]
        ret += abs(v1 - v2) / v1
        n += 1.0
    return ret / n
"""
    if pred.shape[1] == 8:
        return None
    for k in range(pred.shape[0]):
        v1 = label[k]
        v2 = pred[k][0]
        ret += abs(v1 - v2) / v1
        n += 1.0
    return ret / n
"""

vchexing, vcity = build_vocab("price.tsv")
save_vocab(vchexing, "chexing.vocab")
save_vocab(vcity, "city.vocab")

init_c = [('l%d_init_c'%l, (100, 500)) for l in range(2)]
init_h = [('l%d_init_h'%l, (100, 500)) for l in range(2)]
init_states = init_c + init_h

data_train = PriceIter("train.t", 100, vchexing, vcity, init_states)
data_test = PriceIter("test.t", 100, vchexing, vcity, init_states)

def network(seq_len):
    return get_net(chexing_size = len(data_train.chexing_vocab), chexing_embed_size = 500,
                   city_size = len(data_train.city_vocab), city_embed_size = 300, seq_len = seq_len, num_lstm_layer = 2)

devs = [mx.gpu(i) for i in range(1)]
model = mx.model.FeedForward(ctx = devs,
                             symbol = network,
                             num_epoch = 25,
                             learning_rate = 0.0001,
                             wd = 0.0001,
                             lr_scheduler=mx.misc.FactorScheduler(2000,0.9),
                             initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                             momentum = 0.9)


import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

model.fit(X = data_train, eval_data = data_test, eval_metric = mx.metric.np(RMSE), batch_end_callback=mx.callback.Speedometer(32, 50),)

model.save("price-lstm")
