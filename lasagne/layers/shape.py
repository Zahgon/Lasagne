import numpy as np
import theano.tensor as T

from ..theano_extensions import padding
from ..utils import int_types

from .base import Layer


__all__ = [
    "FlattenLayer",
    "flatten",
    "ReshapeLayer",
    "reshape",
    "DimshuffleLayer",
    "dimshuffle",
    "PadLayer",
    "pad",
    "SliceLayer"
]


class FlattenLayer(Layer):
    def __init__(self, incoming, outdim=2, **kwargs):
        super(FlattenLayer, self).__init__(incoming, **kwargs)
        self.outdim = outdim

        if outdim < 1:
            raise ValueError('Dim must be >0, was %i', outdim)

    def get_output_shape_for(self, input_shape):
        to_flatten = input_shape[self.outdim - 1:]

        if any(s is None for s in to_flatten):
            flattened = None
        else:
            flattened = int(np.prod(to_flatten))

        return input_shape[:self.outdim - 1] + (flattened,)

    def get_output_for(self, input, **kwargs):
        return input.flatten(self.outdim)

flatten = FlattenLayer  # shortcut


class ReshapeLayer(Layer):

    def __init__(self, incoming, shape, **kwargs):
        super(ReshapeLayer, self).__init__(incoming, **kwargs)
        shape = tuple(shape)
        for s in shape:
            if isinstance(s, int_types):
                if s == 0 or s < - 1:
                    raise ValueError("`shape` integers must be positive or -1")
            elif isinstance(s, list):
                if len(s) != 1 or not isinstance(s[0], int_types) or s[0] < 0:
                    raise ValueError("`shape` input references must be "
                                     "single-element lists of int >= 0")
            elif isinstance(s, T.TensorVariable):
                if s.ndim != 0:
                    raise ValueError(
                        "A symbolic variable in a shape specification must be "
                        "a scalar, but had %i dimensions" % s.ndim)
            else:
                raise ValueError("`shape` must be a tuple of int and/or [int]")
        if sum(s == -1 for s in shape) > 1:
            raise ValueError("`shape` cannot contain multiple -1")
        self.shape = shape
        self.get_output_shape_for(self.input_shape)

    def get_output_shape_for(self, input_shape, **kwargs):
        output_shape = list(self.shape)
        masked_input_shape = list(input_shape)
        masked_output_shape = list(output_shape)
        for dim, o in enumerate(output_shape):
            if isinstance(o, list):
                if o[0] >= len(input_shape):
                    raise ValueError("specification contains [%d], but input "
                                     "shape has %d dimensions only" %
                                     (o[0], len(input_shape)))
                output_shape[dim] = input_shape[o[0]]
                masked_output_shape[dim] = input_shape[o[0]]
                if (input_shape[o[0]] is None) \
                   and (masked_input_shape[o[0]] is None):
                        masked_input_shape[o[0]] = 1
                        masked_output_shape[dim] = 1
        for dim, o in enumerate(output_shape):
            if isinstance(o, T.TensorVariable):
                output_shape[dim] = None
                masked_output_shape[dim] = None
        input_size = (None if any(x is None for x in masked_input_shape)
                      else np.prod(masked_input_shape))
        output_size = (None if any(x is None for x in masked_output_shape)
                       else np.prod(masked_output_shape))
        del masked_input_shape, masked_output_shape
        if -1 in output_shape:
            dim = output_shape.index(-1)
            if (input_size is None) or (output_size is None):
                output_shape[dim] = None
                output_size = None
            else:
                output_size *= -1
                output_shape[dim] = input_size // output_size
                output_size *= output_shape[dim]
        if (input_size is not None) and (output_size is not None) \
           and (input_size != output_size):
            raise ValueError("%s cannot be reshaped to specification %s. "
                             "The total size mismatches." %
                             (input_shape, self.shape))
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        output_shape = list(self.shape)
        for dim, o in enumerate(output_shape):
            if isinstance(o, list):
                output_shape[dim] = input.shape[o[0]]
        return input.reshape(tuple(output_shape))

reshape = ReshapeLayer  # shortcut


class DimshuffleLayer(Layer):
    def __init__(self, incoming, pattern, **kwargs):
        super(DimshuffleLayer, self).__init__(incoming, **kwargs)

        used_dims = set()
        for p in pattern:
            if isinstance(p, int_types):
                if p in used_dims:
                    raise ValueError("pattern contains dimension {0} more "
                                     "than once".format(p))
                used_dims.add(p)
            elif p == 'x':
                pass
            else:
                raise ValueError("pattern should only contain dimension"
                                 "indices or 'x', not {0}".format(p))

        self.pattern = pattern

        self.get_output_shape_for(self.input_shape)

    def get_output_shape_for(self, input_shape):
        output_shape = []
        dims_used = [False] * len(input_shape)
        for p in self.pattern:
            if isinstance(p, int_types):
                if p < 0 or p >= len(input_shape):
                    raise ValueError("pattern contains {0}, but input shape "
                                     "has {1} dimensions "
                                     "only".format(p, len(input_shape)))
                o = input_shape[p]
                dims_used[p] = True
            elif p == 'x':
                o = 1
            output_shape.append(o)

        for i, (dim_size, used) in enumerate(zip(input_shape, dims_used)):
            if not used and dim_size != 1 and dim_size is not None:
                raise ValueError(
                    "pattern attempted to collapse dimension "
                    "{0} of size {1}; dimensions with size != 1/None are not"
                    "broadcastable and cannot be "
                    "collapsed".format(i, dim_size))

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        return input.dimshuffle(self.pattern)

dimshuffle = DimshuffleLayer  # shortcut


class PadLayer(Layer):
    def __init__(self, incoming, width, val=0, batch_ndim=2, **kwargs):
        super(PadLayer, self).__init__(incoming, **kwargs)
        self.width = width
        self.val = val
        self.batch_ndim = batch_ndim

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        if isinstance(self.width, int_types):
            widths = [self.width] * (len(input_shape) - self.batch_ndim)
        else:
            widths = self.width

        for k, w in enumerate(widths):
            if output_shape[k + self.batch_ndim] is None:
                continue
            else:
                try:
                    l, r = w
                except TypeError:
                    l = r = w
                output_shape[k + self.batch_ndim] += l + r
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        return padding.pad(input, self.width, self.val, self.batch_ndim)

pad = PadLayer  # shortcut


class SliceLayer(Layer):
    def __init__(self, incoming, indices, axis=-1, **kwargs):
        super(SliceLayer, self).__init__(incoming, **kwargs)
        self.slice = indices
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        if isinstance(self.slice, int_types):
            del output_shape[self.axis]
        elif input_shape[self.axis] is not None:
            output_shape[self.axis] = len(
                range(*self.slice.indices(input_shape[self.axis])))
        else:
            output_shape[self.axis] = None
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        axis = self.axis
        if axis < 0:
            axis += input.ndim
        return input[(slice(None),) * axis + (self.slice,)]
