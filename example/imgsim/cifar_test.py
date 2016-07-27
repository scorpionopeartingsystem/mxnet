import sys, random
sys.path.insert(0, '../../python')
import mxnet as mx
import numpy as np
import cifar_sim, cv2
from operator import itemgetter

_, arg_params, __ = mx.model.load_checkpoint("cifar-sim", 100)

batch_size = 1
network = cifar_sim.get_net(batch_size)

input_shapes = dict([('data1', (batch_size, 3, 32, 32)),\
                     ('data2', (batch_size, 3, 32, 32))])
executor = network.simple_bind(ctx = mx.gpu(), **input_shapes)
for key in executor.arg_dict.keys():
    if key in arg_params:
        arg_params[key].copyto(executor.arg_dict[key])

label_imgs = cifar_sim.load_data(['./data/cifar-10-batches-py/test_batch'])

imgs = []

for la, tmp in label_imgs.items():
    for img in tmp:
        imgs.append((img, la))

random.shuffle(imgs)

def save_img(fname, im):
    a = np.copy(im) * 255.0
    cv2.imwrite(fname, a.transpose(1, 2, 0))

src = imgs[0][0].reshape((3, 32, 32))
src_label = imgs[0][1]
save_img("src.png", src)
ret = []
for i in range(1000):
    k = random.randint(0, len(imgs) - 1)
    dst = imgs[k][0].reshape((3, 32, 32))
    executor.forward(is_train = True, data1 = mx.nd.array([src]),
                     data2 = mx.nd.array([dst]))
    probs = executor.outputs[0].asnumpy()
    ret.append((dst, probs[0][1], imgs[k][1]))

i = 0
print src_label
for img, w, l in sorted(ret, key = itemgetter(1), reverse = True)[:5]:
    print w, l
    save_img("dst_" + str(i) + ".png", img)
    i += 1
