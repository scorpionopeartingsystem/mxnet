# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx
from io import BytesIO
from captcha.image import ImageCaptcha
import cv2, random

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

def gen_rand():
    num = random.randint(0, 9999)
    buf = str(num)
    while len(buf) < 4:
        buf = "0" + buf
    return buf

def get_label(buf):
    ret = np.zeros((len(buf), 10))
    k = 0
    for c in buf:
        c = int(c)
        ret[k][c] = 1.
        k += 1
    return k

class OCRIter(mx.io.DataIter):
    def __init__(self, batch_size,
                 data_name='data', label_name='label'):
        super(OCRIter, self).__init__()
        self.captcha = ImageCaptcha(fonts=['/Library/Fonts/Skia.ttf'])
        self.batch_size = batch_size

        self.provide_data = [('data', (batch_size))]
        self.provide_label = [('softmax_label', (self.batch_size))]

    def __iter__(self):
        data = []
        label = []
        for i in range(self.batch_size):
            num = gen_rand()
            img = self.captcha.generate(num)
            img = np.fromstring(data.getvalue(), dtype='uint8')
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            data.append(img)
            label.append(get_label(num))

        data_all = [mx.nd.array(data)]
        label_all = [mx.nd.array(label)]
        data_names = ['data']
        label_names = ['softmax_label']

        data_batch = SimpleBatch(data_names, data_all, label_names, label_all,
                                 self.buckets[i_bucket])
        yield data_batch

    def reset(self):
        pass
