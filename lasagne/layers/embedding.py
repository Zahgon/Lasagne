import numpy as np
import theano.tensor as T

from .. import init
from .base import Layer


__all__ = [
    "EmbeddingLayer"
]


class EmbeddingLayer(Layer):
    def __init__(self, incoming, input_size, output_size,
                 W=init.Normal(), **kwargs):
        super(EmbeddingLayer, self).__init__(incoming, **kwargs)

        self.input_size = input_size
        self.output_size = output_size

        self.W = self.add_param(W, (input_size, output_size), name="W")

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size, )

    def get_output_for(self, input, **kwargs):
        return self.W[input]
