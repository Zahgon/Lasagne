# -*- coding: utf-8 -*-

import theano.tensor


def sigmoid(x):
    pass


def softmax(x):
    pass


def tanh(x):
    pass


class ScaledTanH(object):

    def __init__(self, scale_in=1, scale_out=1):
        self.scale_in = scale_in
        self.scale_out = scale_out

    def __call__(self, x):
        return theano.tensor.tanh(x * self.scale_in) * self.scale_out


ScaledTanh = ScaledTanH  # alias with alternative capitalization


def rectify(x):
    pass


class LeakyRectify(object):
    def __init__(self, leakiness=0.01):
        self.leakiness = leakiness

    def __call__(self, x):
        return theano.tensor.nnet.relu(x, self.leakiness)


leaky_rectify = LeakyRectify()  # shortcut with default leakiness
leaky_rectify.__doc__ = """leaky_rectify(x)

    Instance of :class:`LeakyRectify` with leakiness :math:`\\alpha=0.01`
    """


very_leaky_rectify = LeakyRectify(1./3)  # shortcut with high leakiness
very_leaky_rectify.__doc__ = """very_leaky_rectify(x)

     Instance of :class:`LeakyRectify` with leakiness :math:`\\alpha=1/3`
     """


def elu(x):
    pass


class SELU(object):
    def __init__(self, scale=1, scale_neg=1):
        self.scale = scale
        self.scale_neg = scale_neg

    def __call__(self, x):
        return self.scale * theano.tensor.switch(
                x > 0.0,
                x,
                self.scale_neg * (theano.tensor.expm1(x)))


selu = SELU(scale=1.0507009873554804934193349852946,
            scale_neg=1.6732632423543772848170429916717)
selu.__doc__ = """selu(x)

    Instance of :class:`SELU` with :math:`\\alpha\\approx 1.6733,
    \\lambda\\approx 1.0507`

    This has a stable and attracting fixed point of :math:`\\mu=0`,
    :math:`\\sigma=1` under the assumptions of the
    original paper on self-normalizing neural networks.
    """


def softplus(x):
    pass


def linear(x):
    pass

identity = linear
