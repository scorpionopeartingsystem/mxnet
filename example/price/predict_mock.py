# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx
from train_mock import get_net

batch_size = 1
data_shape = [("dis", (batch_size, 1)), ("series", (batch_size, 1))]
input_shapes = dict(data_shape)

_, arg_params, __ = mx.model.load_checkpoint("mock", 10)

net = get_net()
executor = net.simple_bind(ctx=mx.cpu(), **input_shapes)

for key in executor.arg_dict.keys():
    if key in arg_params:
        arg_params[key].copyto(executor.arg_dict[key])
        

dis = mx.nd.array([np.array([float(sys.argv[1]) / 10.0])])
s = mx.nd.array([np.array([int(sys.argv[2])], dtype=np.int)])

executor.forward(is_train = False, dis = dis, series = s)

for o in executor.outputs:
    print o.asnumpy()
