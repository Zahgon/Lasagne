import warnings

import theano

from .. import init
from .. import nonlinearities
from .base import Layer

from .conv import conv_output_length, BaseConvLayer
from .pool import pool_output_length
from .normalization import BatchNormLayer
from ..utils import as_tuple

try:
    from theano import gpuarray as gpu
except ImportError:
    from theano.sandbox import gpuarray as gpu
gpu_enabled = gpu.pygpu_activated
dnn_enabled = gpu.dnn.dnn_present
if not gpu_enabled:
    try:
        from theano.sandbox import cuda as gpu
        import theano.sandbox.cuda.dnn
    except Exception:  # Theano 0.10+ raises nose.SkipTest
        gpu_enabled = False
    else:
        gpu_enabled = gpu.cuda_enabled
        dnn_enabled = gpu.dnn.dnn_available
if gpu_enabled:
    if dnn_enabled():
        dnn = gpu.dnn
    else:
        raise ImportError(
            "cuDNN not available: %s\nSee http://lasagne.readthedocs.org\
            /en/latest/user/installation.html#cudnn\
            " % dnn_enabled.msg)  # pragma: no cover
else:
    raise ImportError(
        "requires GPU support -- see http://lasagne.readthedocs.org/en/"
        "latest/user/installation.html#gpu-support")  # pragma: no cover

if theano.config.floatX == 'float64':
    warnings.warn("You are using a GPU layer with Theano configured for "
                  "double precision (floatX=float64). Depending on your "
                  "Theano version and GPU, this may be slow or unsupported."
                  "We recommend to configure Theano for single precision "
                  "(floatX=float32); see http://lasagne.readthedocs.org/en/"
                  "latest/user/installation.html#gpu-support.")

__all__ = [
    "Pool2DDNNLayer",
    "MaxPool2DDNNLayer",
    "Pool3DDNNLayer",
    "MaxPool3DDNNLayer",
    "Conv2DDNNLayer",
    "Conv3DDNNLayer",
    "SpatialPyramidPoolingDNNLayer",
    "BatchNormDNNLayer",
    "batch_norm_dnn",
]


class Pool2DDNNLayer(Layer):
    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, mode='max', **kwargs):
        super(Pool2DDNNLayer, self).__init__(incoming, **kwargs)
        if len(self.input_shape) != 4:
            raise ValueError("Tried to create a 2D pooling layer with "
                             "input shape %r. Expected 4 input dimensions "
                             "(batchsize, channels, 2 spatial dimensions)."
                             % (self.input_shape,))
        self.pool_size = as_tuple(pool_size, 2)
        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 2)
        self.pad = as_tuple(pad, 2)
        self.mode = mode
        if not ignore_border:
            raise NotImplementedError("Pool2DDNNLayer does not support "
                                      "ignore_border=False.")

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad[0],
                                             ignore_border=True,
                                             )

        output_shape[3] = pool_output_length(input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad[1],
                                             ignore_border=True,
                                             )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        return dnn.dnn_pool(input, self.pool_size, self.stride,
                            self.mode, self.pad)


class MaxPool2DDNNLayer(Pool2DDNNLayer):
    def __init__(self, incoming, pool_size, stride=None,
                 pad=(0, 0), ignore_border=True, **kwargs):
        super(MaxPool2DDNNLayer, self).__init__(incoming, pool_size, stride,
                                                pad, ignore_border, mode='max',
                                                **kwargs)


class Pool3DDNNLayer(Layer):
    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0, 0),
                 ignore_border=True, mode='max', **kwargs):
        super(Pool3DDNNLayer, self).__init__(incoming, **kwargs)
        if len(self.input_shape) != 5:
            raise ValueError("Tried to create a 3D pooling layer with "
                             "input shape %r. Expected 5 input dimensions "
                             "(batchsize, channels, 3 spatial dimensions)."
                             % (self.input_shape,))
        self.pool_size = as_tuple(pool_size, 3)
        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 3)
        self.pad = as_tuple(pad, 3)
        self.mode = mode
        if not ignore_border:
            raise NotImplementedError("Pool3DDNNLayer does not support "
                                      "ignore_border=False.")

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad[0],
                                             ignore_border=True,
                                             )

        output_shape[3] = pool_output_length(input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad[1],
                                             ignore_border=True,
                                             )

        output_shape[4] = pool_output_length(input_shape[4],
                                             pool_size=self.pool_size[2],
                                             stride=self.stride[2],
                                             pad=self.pad[2],
                                             ignore_border=True,
                                             )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        return dnn.dnn_pool(input, self.pool_size, self.stride,
                            self.mode, self.pad)


class MaxPool3DDNNLayer(Pool3DDNNLayer):
    def __init__(self, incoming, pool_size, stride=None,
                 pad=(0, 0, 0), ignore_border=True, **kwargs):
        super(MaxPool3DDNNLayer, self).__init__(incoming, pool_size, stride,
                                                pad, ignore_border, mode='max',
                                                **kwargs)


class Conv2DDNNLayer(BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 flip_filters=False, num_groups=1, **kwargs):
        super(Conv2DDNNLayer, self).__init__(incoming, num_filters,
                                             filter_size, stride, pad,
                                             untie_biases, W, b, nonlinearity,
                                             flip_filters, num_groups, n=2,
                                             **kwargs)

    def convolve(self, input, **kwargs):
        conv_mode = 'conv' if self.flip_filters else 'cross'
        border_mode = self.pad
        if border_mode == 'same':
            border_mode = tuple(s // 2 for s in self.filter_size)
        extra_kwargs = {}
        if self.num_groups > 1:  # pragma: no cover
            extra_kwargs = {'num_groups': self.num_groups}

        conved = dnn.dnn_conv(img=input,
                              kerns=self.W,
                              subsample=self.stride,
                              border_mode=border_mode,
                              conv_mode=conv_mode,
                              **extra_kwargs)
        return conved


class Conv3DDNNLayer(BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1, 1),
                 pad=0, untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 flip_filters=False, num_groups=1, **kwargs):
        super(Conv3DDNNLayer, self).__init__(incoming, num_filters,
                                             filter_size, stride, pad,
                                             untie_biases, W, b, nonlinearity,
                                             flip_filters, num_groups, n=3,
                                             **kwargs)

    def convolve(self, input, **kwargs):
        conv_mode = 'conv' if self.flip_filters else 'cross'
        border_mode = self.pad
        if border_mode == 'same':
            border_mode = tuple(s // 2 for s in self.filter_size)
        extra_kwargs = {}
        if self.num_groups > 1:
            extra_kwargs = {'num_groups': self.num_groups}

        conved = dnn.dnn_conv3d(img=input,
                                kerns=self.W,
                                subsample=self.stride,
                                border_mode=border_mode,
                                conv_mode=conv_mode,
                                **extra_kwargs)
        return conved


class SpatialPyramidPoolingDNNLayer(Layer):
    def __init__(self, incoming, pool_dims=[4, 2, 1], mode='max', **kwargs):
            super(SpatialPyramidPoolingDNNLayer, self).__init__(incoming,
                                                                **kwargs)
            if len(self.input_shape) != 4:
                raise ValueError("Tried to create a SPP layer with "
                                 "input shape %r. Expected 4 input dimensions "
                                 "(batchsize, channels, 2 spatial dimensions)."
                                 % (self.input_shape,))
            self.mode = mode
            self.pool_dims = pool_dims

    def get_output_for(self, input, **kwargs):
        input_size = tuple(symb if fixed is None else fixed
                           for fixed, symb
                           in zip(self.input_shape[2:], input.shape[2:]))
        pool_list = []
        for pool_dim in self.pool_dims:
            win_size = tuple((i + pool_dim - 1) // pool_dim
                             for i in input_size)
            str_size = tuple(i // pool_dim for i in input_size)

            pool = dnn.dnn_pool(input, win_size, str_size, self.mode, (0, 0))
            pool = pool.flatten(3)
            pool_list.append(pool)

        return theano.tensor.concatenate(pool_list, axis=2)

    def get_output_shape_for(self, input_shape):
        num_features = sum(p*p for p in self.pool_dims)
        return (input_shape[0], input_shape[1], num_features)


class BatchNormDNNLayer(BatchNormLayer):
    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), **kwargs):
        super(BatchNormDNNLayer, self).__init__(
                incoming, axes, epsilon, alpha, beta, gamma, mean, inv_std,
                **kwargs)
        all_but_second_axis = (0,) + tuple(range(2, len(self.input_shape)))
        if self.axes not in ((0,), all_but_second_axis):
            raise ValueError("BatchNormDNNLayer only supports normalization "
                             "across the first axis, or across all but the "
                             "second axis, got axes=%r" % (axes,))

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]
        unpattern = [d for d in range(input.ndim) if d not in self.axes]

        if not use_averages or update_averages:
            shape = tuple(s for (d, s) in enumerate(input.shape)
                          if d not in self.axes)
            gamma = self.gamma or theano.tensor.ones(shape)
            beta = self.beta or theano.tensor.zeros(shape)
            mode = 'per-activation' if self.axes == (0,) else 'spatial'
            (normalized,
             input_mean,
             input_inv_std) = dnn.dnn_batch_normalization_train(
                    input, gamma.dimshuffle(pattern), beta.dimshuffle(pattern),
                    mode, self.epsilon)

        if use_averages:
            mean = self.mean.dimshuffle(pattern)
            inv_std = self.inv_std.dimshuffle(pattern)
            gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
            beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
            normalized = (input - mean) * (gamma * inv_std) + beta

        if update_averages:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean.dimshuffle(unpattern))
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std.dimshuffle(unpattern))
            dummy = 0 * (running_mean + running_inv_std).dimshuffle(pattern)
            normalized = normalized + dummy

        return normalized


def batch_norm_dnn(layer, **kwargs):
    pass


if not hasattr(dnn, 'dnn_batch_normalization_train'):
    del BatchNormDNNLayer, batch_norm_dnn
    __all__.remove('BatchNormDNNLayer')
    __all__.remove('batch_norm_dnn')
