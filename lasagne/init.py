
import numpy as np

from .utils import floatX
from .random import get_rng


class Initializer(object):
    def __call__(self, shape):
        """
        Makes :class:`Initializer` instances callable like a function, invoking
        their :meth:`sample()` method.
        """
        return self.sample(shape)

    def sample(self, shape):
        """
        Sample should return a theano.tensor of size shape and data type
        theano.config.floatX.

        Parameters
        -----------
        shape : tuple or int
            Integer or tuple specifying the size of the returned
            matrix.
        returns : theano.tensor
            Matrix of size shape and dtype theano.config.floatX.
        """
        raise NotImplementedError()


class Normal(Initializer):
    def __init__(self, std=0.01, mean=0.0):
        self.std = std
        self.mean = mean



class Uniform(Initializer):
    def __init__(self, range=0.01, std=None, mean=0.0):
        if std is not None:
            a = mean - np.sqrt(3) * std
            b = mean + np.sqrt(3) * std
        else:
            try:
                a, b = range  # range is a tuple
            except TypeError:
                a, b = -range, range  # range is a number

        self.range = (a, b)



class Glorot(Initializer):
    def __init__(self, initializer, gain=1.0, c01b=False):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.initializer = initializer
        self.gain = gain
        self.c01b = c01b



class GlorotNormal(Glorot):
    def __init__(self, gain=1.0, c01b=False):
        super(GlorotNormal, self).__init__(Normal, gain, c01b)


class GlorotUniform(Glorot):
    def __init__(self, gain=1.0, c01b=False):
        super(GlorotUniform, self).__init__(Uniform, gain, c01b)


class He(Initializer):
    def __init__(self, initializer, gain=1.0, c01b=False):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.initializer = initializer
        self.gain = gain
        self.c01b = c01b



class HeNormal(He):
    def __init__(self, gain=1.0, c01b=False):
        super(HeNormal, self).__init__(Normal, gain, c01b)


class HeUniform(He):
    def __init__(self, gain=1.0, c01b=False):
        super(HeUniform, self).__init__(Uniform, gain, c01b)


class Constant(Initializer):
    def __init__(self, val=0.0):
        self.val = val



class Sparse(Initializer):
    def __init__(self, sparsity=0.1, std=0.01):
        self.sparsity = sparsity
        self.std = std



class Orthogonal(Initializer):
    def __init__(self, gain=1.0):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.gain = gain

