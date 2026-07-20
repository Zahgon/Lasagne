import warnings

import numpy as np
import theano
import theano.tensor as T

from .. import init
from .. import nonlinearities

from .base import Layer

from .conv import conv_output_length, BaseConvLayer
from .pool import pool_output_length
from ..utils import as_tuple

try:
    from theano import gpuarray
except ImportError:
    from theano.sandbox import gpuarray
if gpuarray.pygpu_activated:
    raise ImportError("cuda_convnet is not supported under Theano's new "
                      "GPU backend. Please either use the ordinary "
                      "convolutional and pooling layers (they are usually "
                      "faster), or use Theano's old GPU backend (available "
                      "up to Theano 0.9).")  # pragma: no cover

from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs

__all__ = [
    "Conv2DCCLayer",
    "MaxPool2DCCLayer",
    "ShuffleBC01ToC01BLayer",
    "bc01_to_c01b",
    "ShuffleC01BToBC01Layer",
    "c01b_to_bc01",
    "NINLayer_c01b",
]


if not theano.sandbox.cuda.cuda_enabled:
    raise ImportError(
            "requires GPU support -- see http://lasagne.readthedocs.org/en/"
            "latest/user/installation.html#gpu-support")  # pragma: no cover

if theano.config.floatX == 'float64':
    warnings.warn("You are using a GPU layer with Theano configured for "
                  "double precision (floatX=float64). Depending on your "
                  "Theano version and GPU, this may be slow or unsupported. "
                  "We recommend to configure Theano for single precision "
                  "(floatX=float32); see http://lasagne.readthedocs.org/en/"
                  "latest/user/installation.html#gpu-support.")


class Conv2DCCLayer(BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False, W=None,
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 dimshuffle=True, flip_filters=False, partial_sum=1,
                 **kwargs):
        if W is None:
            if dimshuffle:
                W = init.GlorotUniform()
            else:
                W = init.GlorotUniform(c01b=True)
        self.dimshuffle = dimshuffle

        super(Conv2DCCLayer, self).__init__(incoming, num_filters, filter_size,
                                            stride, pad, untie_biases, W, b,
                                            nonlinearity, flip_filters, n=2,
                                            **kwargs)
        self.partial_sum = partial_sum

        if self.filter_size[0] != self.filter_size[1]:
            raise RuntimeError("Conv2DCCLayer only supports square filters, "
                               "but filter_size=(%d, %d)" % filter_size)

        if self.stride[0] != self.stride[1]:
            raise RuntimeError("Conv2DCCLayer only supports square strides, "
                               "but stride=(%d, %d)" % stride)

        if self.num_filters % 16 != 0:
            raise RuntimeError("Conv2DCCLayer requires num_filters to be a "
                               "multiple of 16, but num_filters is "
                               "%d" % num_filters)

        if not (self.num_input_channels < 4 or
                self.num_input_channels % 4 == 0):
            raise RuntimeError("Conv2DCCLayer requires the number of input "
                               "channels to be 1, 2, 3 or a multiple of 4, "
                               "but it is %d" % self.num_input_channels)

        if isinstance(self.pad, tuple):
            if self.pad[0] != self.pad[1]:
                raise RuntimeError("Conv2DCCLayer only supports square "
                                   "padding, but pad=(%d, %d)" % pad)
            pad = self.pad[0]
        elif self.pad == 'same':
            pad = self.filter_size[0] // 2
        elif self.pad == 'full':
            pad = self.filter_size[0] - 1

        if not self.dimshuffle and self.untie_biases and self.b is not None:
            del self.params[self.b]
            biases_shape = (num_filters, self.output_shape[1],
                            self.output_shape[2])
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

        self.filter_acts_op = FilterActs(stride=self.stride[0],
                                         partial_sum=self.partial_sum,
                                         pad=pad)


    def get_W_shape(self):
        if self.dimshuffle:
            return super(Conv2DCCLayer, self).get_W_shape()
        else:
            return ((self.num_input_channels,) +
                    self.filter_size +
                    (self.num_filters,))

    def get_output_shape_for(self, input_shape):
        if not self.dimshuffle:
            input_shape = (input_shape[3], input_shape[0],
                           input_shape[1], input_shape[2])
        shape = super(Conv2DCCLayer, self).get_output_shape_for(input_shape)
        if not self.dimshuffle:
            shape = (shape[1], shape[2], shape[3], shape[0])
        return shape

    def get_output_for(self, input, **kwargs):
        if self.dimshuffle:
            filters = self.W.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
            input = input.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
        else:
            filters = self.W

        if self.flip_filters:
            filters = filters[:, ::-1, ::-1, :]  # flip top-down, left-right

        contiguous_filters = gpu_contiguous(filters)
        contiguous_input = gpu_contiguous(input)
        conved = self.filter_acts_op(contiguous_input, contiguous_filters)

        if self.stride != 1:
            pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * 2
            true_rows = conv_output_length(input.shape[1],
                                           self.filter_size[0],
                                           self.stride[0],
                                           pad[0])
            true_columns = conv_output_length(input.shape[2],
                                              self.filter_size[1],
                                              self.stride[1],
                                              pad[1])
            conved = conved[:, :true_rows, :true_columns, :]

        if self.b is not None:
            if self.untie_biases:
                biases = self.b.dimshuffle(0, 1, 2, 'x')  # c01 to c01b
            else:
                biases = self.b.dimshuffle(0, 'x', 'x', 'x')  # c to c01b
            conved += biases

        conved = self.nonlinearity(conved)

        if self.dimshuffle:
            return conved.dimshuffle(3, 0, 1, 2)  # c01b to bc01
        else:
            return conved


class MaxPool2DCCLayer(Layer):
    def __init__(self, incoming, pool_size, stride=None, ignore_border=False,
                 dimshuffle=True, **kwargs):
        from pylearn2.sandbox.cuda_convnet.pool import MaxPool

        if 'pad' in kwargs:
            pad = kwargs.pop('pad')
            if as_tuple(pad, 2) != (0, 0):
                raise NotImplementedError("MaxPool2DCCLayer does not "
                                          "support padding")

        super(MaxPool2DCCLayer, self).__init__(incoming, **kwargs)

        pool_size = as_tuple(pool_size, 2)

        if pool_size[0] != pool_size[1]:
            raise NotImplementedError("MaxPool2DCCLayer only supports square "
                                      "pooling regions, but pool_size=(%d, %d)"
                                      % pool_size)

        self.pool_size = pool_size[0]

        if stride is None:
            self.stride = self.pool_size
        else:
            stride = as_tuple(stride, 2)
            if stride[0] != stride[1]:
                raise NotImplementedError("MaxPool2DCCLayer only supports "
                                          "using the same stride in both "
                                          "directions but stride=(%d, %d)"
                                          % stride)
            self.stride = stride[0]

        if self.stride > self.pool_size:
            raise NotImplementedError("MaxPool2DCCLayer only supports "
                                      "stride <= pool_size.")

        if ignore_border:
            raise NotImplementedError("MaxPool2DCCLayer does not support "
                                      "ignore_border=True.")

        self.dimshuffle = dimshuffle

        self.pool_op = MaxPool(ds=self.pool_size, stride=self.stride)

    def get_output_shape_for(self, input_shape):
        if self.dimshuffle:
            batch_size = input_shape[0]
            num_input_channels = input_shape[1]
            input_rows, input_columns = input_shape[2:4]
        else:
            batch_size = input_shape[3]
            num_input_channels = input_shape[0]
            input_rows, input_columns = input_shape[1:3]

        output_rows = pool_output_length(input_rows,
                                         pool_size=self.pool_size,
                                         stride=self.stride,
                                         pad=0,
                                         ignore_border=False,
                                         )
        output_columns = pool_output_length(input_columns,
                                            pool_size=self.pool_size,
                                            stride=self.stride,
                                            pad=0,
                                            ignore_border=False,
                                            )

        if self.dimshuffle:
            return (batch_size, num_input_channels, output_rows,
                    output_columns)
        else:
            return (num_input_channels, output_rows, output_columns,
                    batch_size)

    def get_output_for(self, input, **kwargs):
        if self.dimshuffle:
            input = input.dimshuffle(1, 2, 3, 0)  # bc01 to c01b

        contiguous_input = gpu_contiguous(input)
        pooled = self.pool_op(contiguous_input)

        if self.dimshuffle:
            return pooled.dimshuffle(3, 0, 1, 2)  # c01b to bc01
        else:
            return pooled



class ShuffleBC01ToC01BLayer(Layer):
    def get_output_shape_for(self, input_shape):
        return (input_shape[1], input_shape[2], input_shape[3], input_shape[0])

    def get_output_for(self, input, **kwargs):
        return input.dimshuffle(1, 2, 3, 0)

bc01_to_c01b = ShuffleBC01ToC01BLayer  # shortcut


class ShuffleC01BToBC01Layer(Layer):
    def get_output_shape_for(self, input_shape):
        return (input_shape[3], input_shape[0], input_shape[1], input_shape[2])

    def get_output_for(self, input, **kwargs):
        return input.dimshuffle(3, 0, 1, 2)

c01b_to_bc01 = ShuffleC01BToBC01Layer  # shortcut



class NINLayer_c01b(Layer):
    def __init__(self, incoming, num_units, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, **kwargs):
        super(NINLayer_c01b, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = num_units
        self.untie_biases = untie_biases

        num_input_channels = self.input_shape[0]

        self.W = self.add_param(W, (num_units, num_input_channels), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_units,) + self.output_shape[1:-1]
            else:
                biases_shape = (num_units,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (self.num_units,) + input_shape[1:]

    def get_output_for(self, input, **kwargs):
        out = T.tensordot(self.W, input, axes=[[1], [0]])

        if self.b is None:
            activation = out
        else:
            if self.untie_biases:
                bias_axes = range(input.ndim - 1) + ['x']
            else:
                bias_axes = [0] + (['x'] * (input.ndim - 1))
            b_shuffled = self.b.dimshuffle(bias_axes)
            activation = out + b_shuffled

        return self.nonlinearity(activation)
