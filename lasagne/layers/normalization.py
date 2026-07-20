# -*- coding: utf-8 -*-


import theano
import theano.tensor as T

from .. import init
from .. import nonlinearities
from ..utils import int_types

from .base import Layer


__all__ = [
    "LocalResponseNormalization2DLayer",
    "BatchNormLayer",
    "batch_norm",
    "StandardizationLayer",
    "instance_norm",
    "layer_norm",
]


class LocalResponseNormalization2DLayer(Layer):

    def __init__(self, incoming, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        super(LocalResponseNormalization2DLayer, self).__init__(incoming,
                                                                **kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n")

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        input_shape = self.input_shape
        if any(s is None for s in input_shape):
            input_shape = input.shape
        half_n = self.n // 2
        input_sqr = T.sqr(input)
        b, ch, r, c = input_shape
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],
                                    input_sqr)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        return input / scale


class BatchNormLayer(Layer):
    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int_types):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha

        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, shape, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, shape, 'gamma',
                                        trainable=True, regularizable=True)
        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False,
                                   batch_norm_stat=True)
        self.inv_std = self.add_param(inv_std, shape, 'inv_std',
                                      trainable=False, regularizable=False,
                                      batch_norm_stat=True)

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):
        input_mean = input.mean(self.axes)
        input_inv_std = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))

        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        if use_averages:
            mean = self.mean
            inv_std = self.inv_std
        else:
            mean = input_mean
            inv_std = input_inv_std

        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        if update_averages:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std)
            mean += 0 * running_mean
            inv_std += 0 * running_inv_std

        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        inv_std = inv_std.dimshuffle(pattern)

        normalized = (input - mean) * (gamma * inv_std) + beta
        return normalized


def batch_norm(layer, **kwargs):
    pass


class StandardizationLayer(Layer):
    def __init__(self, incoming, axes='auto', epsilon=1e-4, **kwargs):
        super(StandardizationLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            if len(self.input_shape) == 2:
                axes = (1,)
            else:
                axes = tuple(range(2, len(self.input_shape)))
        elif axes == 'spatial':
            axes = tuple(range(2, len(self.input_shape)))
        elif axes == 'features':
            axes = tuple(range(1, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon

    def get_output_for(self, input, **kwargs):
        mean = input.mean(self.axes, keepdims=True)
        std = T.sqrt(input.var(self.axes, keepdims=True) + self.epsilon)
        return (input - mean) / std


def instance_norm(layer, learn_scale=True, learn_bias=True, **kwargs):
    pass


def layer_norm(layer, **kwargs):
    pass
