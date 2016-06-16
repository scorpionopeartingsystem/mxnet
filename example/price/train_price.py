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

def process_line(line):
    price, create_time, car_id, minor_category_name, \
        license_date, license_month, road_haul, city_id, \
        pinpai, chexi, chexing, guobie, zhidaojiage, \
        niankuan, cheliangleixing, cheliangjibie, biansuxiangleixing, \
        cheshenxingshi = line.strip('\n').split('\t')
    price = float(price)
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
    return price / 10000.0, vehicle_age, 'pp_' + pinpai, 'chexi_' + chexi, \
        'chexing_' + chexing, road_haul, city_id, \
        zhidaojiage_min, zhidaojiage_max, 'gb_' + guobie, \
        'nk_' + niankuan, 'cllx_' + cheliangleixing, 'cljb_' + cheliangjibie, \
        'bsxlx_' + biansuxiangleixing, 'csxs_' + cheshenxingshi

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
    def __init__(self, fname, batch_size, vchexing, vcity, is_train):
        super(PriceIter, self).__init__()
        self.batch_size = batch_size
        self.is_train = is_train
        self.city = []
        self.chexing = []
        self.data = []
        self.chexing_vocab = vchexing
        self.city_vocab = vcity
        self.price = []
        n = 0
        for line in file(fname):
            n += 1
            if is_train and n % 10 == 0: 
                continue
            if not is_train and n % 10 > 0:
                continue
            price, vehicle_age, pinpai, chexi, chexing, road_haul, city_id, \
                zhidaojiage_min, zhidaojiage_max, guobie, niankuan, cheliangleixing, cheliangjibie, biansuxiangleixing, cheshenxingshi = process_line(line)
            self.data.append(np.array([vehicle_age / 10.0, float(road_haul)/ 10.0, 
                                        zhidaojiage_min / 10.0, zhidaojiage_max / 10.0]))
            self.chexing.append(np.array([self.chexing_vocab[pinpai], 
                                          self.chexing_vocab[chexi], 
                                          self.chexing_vocab[chexing],
                                          self.chexing_vocab[guobie],
                                          self.chexing_vocab[niankuan],
                                          self.chexing_vocab[cheliangleixing],
                                          self.chexing_vocab[cheliangjibie],
                                          self.chexing_vocab[biansuxiangleixing],
                                          self.chexing_vocab[cheshenxingshi]
                                      ], dtype=np.int))
            
            self.city.append(np.array([self.city_vocab[city_id]], dtype=np.int))
            self.price.append(price)

        self.provide_data = [('data', (batch_size, 4)),
                             ('chexing', (batch_size, 9)),
                             ('city', (batch_size, 1))]
        self.provide_label = [('price', (self.batch_size, ))]

    def __iter__(self):
        count = len(self.price)
        for i in range(count / self.batch_size):
            bdata = []
            bcity = []
            bchexing = []
            blabel = []
            blabel_interval = []
            for j in range(self.batch_size):
                k = (i * self.batch_size + j)
                bdata.append(self.data[k])
                bcity.append(self.city[k])
                bchexing.append(self.chexing[k])
                blabel.append(self.price[k])
                
            data_all = [mx.nd.array(bdata),
                        mx.nd.array(bchexing),
                        mx.nd.array(bcity)
            ]
            label_all = [mx.nd.array(blabel)]
            data_names = ['data', 'chexing', 'city']
            label_names = ['price']
            
            data_batch = Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

def get_net(chexing_size, chexing_embed_size,
            city_size, city_embed_size):
    data = mx.symbol.Variable('data')
    price = mx.symbol.Variable('price')
    chexing = mx.symbol.Variable('chexing')
    city = mx.symbol.Variable('city')

    data = mx.symbol.Flatten(data = data)
    chexing_embed = mx.symbol.Embedding(data = chexing, 
                                        input_dim = chexing_size,
                                        output_dim = chexing_embed_size,
                                        name="chexing_embed")
    chexing_flatten = mx.symbol.Flatten(chexing_embed)

    city_embed = mx.symbol.Embedding(data = city, input_dim = city_size, 
                                     output_dim = city_embed_size,
                                     name="city_embed")
    city_flatten = mx.symbol.Flatten(city_embed)

    concat = mx.symbol.Concat(*[data,
                             chexing_flatten, city_flatten], dim = 1)
    rnet = mx.symbol.FullyConnected(data = concat, num_hidden = 100, name="rfc1")
    rnet = mx.symbol.Activation(data = rnet, act_type="relu")
    rnet = mx.symbol.FullyConnected(data = rnet, num_hidden = 100, name= "rfc2")
    rnet = mx.symbol.Activation(data = rnet, act_type="relu")
    rnet = mx.symbol.FullyConnected(data = rnet, num_hidden = 1, name = "rfc3")
    rnet = mx.symbol.Activation(data = rnet, act_type="relu")
    rnet = mx.symbol.LinearRegressionOutput(data = rnet, label = price, name="rnet")
    return rnet

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

fname = sys.argv[1]
vchexing, vcity = build_vocab("./data/price.tsv")
save_vocab(vchexing, "./data/chexing.vocab")
save_vocab(vcity, "./data/city.vocab")

data_train = PriceIter("./data/" + fname + ".tsv", 32, vchexing, vcity, True)
data_test = PriceIter("./data/" + fname + ".tsv", 32, vchexing, vcity, False)

network = get_net(chexing_size = len(data_train.chexing_vocab), 
                  chexing_embed_size = 1000,
                  city_size = len(data_train.city_vocab), city_embed_size = 300)

devs = [mx.gpu(i) for i in range(1)]
model = mx.model.FeedForward(ctx = devs,
                             symbol = network,
                             num_epoch = 25,
                             learning_rate = 0.0001,
                             wd = 0.0001,
                             initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                             momentum = 0.9)


import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

model.fit(X = data_train, eval_data = data_test, 
          eval_metric = mx.metric.np(RMSE), 
          batch_end_callback=mx.callback.Speedometer(32, 50),)

model.save("./data/" + fname)
