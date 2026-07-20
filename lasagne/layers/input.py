from collections import OrderedDict

import theano
import theano.tensor as T

from .. import utils

from .base import Layer


__all__ = [
    "InputLayer",
]


class InputLayer(Layer):
    def __init__(self, shape, input_var=None, name=None, **kwargs):
        self.shape = tuple(shape)
        if any(d is not None and d <= 0 for d in self.shape):
            raise ValueError((
                "Cannot create InputLayer with a non-positive shape "
                "dimension. shape=%r, self.name=%r") % (
                    self.shape, name))

        ndim = len(self.shape)
        if input_var is None:
            input_var_type = T.TensorType(theano.config.floatX,
                                          [s == 1 for s in self.shape])
            var_name = ("%s.input" % name) if name is not None else "input"
            input_var = input_var_type(var_name)
        else:
            if input_var.ndim != ndim:
                raise ValueError("shape has %d dimensions, but variable has "
                                 "%d" % (ndim, input_var.ndim))
        self.input_var = input_var
        self.name = name
        self.params = OrderedDict()

