import theano
import theano.tensor as T

from .base import Layer
from ..random import get_rng

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


__all__ = [
    "DropoutLayer",
    "dropout",
    "dropout_channels",
    "spatial_dropout",
    "dropout_locations",
    "GaussianNoiseLayer",
]


class DropoutLayer(Layer):
    def __init__(self, incoming, p=0.5, rescale=True, shared_axes=(),
                 **kwargs):
        super(DropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p
        self.rescale = rescale
        self.shared_axes = tuple(shared_axes)

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic or self.p == 0:
            return input
        else:
            one = T.constant(1, dtype='int8')

            retain_prob = one - self.p
            if self.rescale:
                input /= retain_prob

            mask_shape = self.input_shape
            if any(s is None for s in mask_shape):
                mask_shape = input.shape

            if self.shared_axes:
                shared_axes = tuple(a if a >= 0 else a + input.ndim
                                    for a in self.shared_axes)
                mask_shape = tuple(1 if a in shared_axes else s
                                   for a, s in enumerate(mask_shape))
            mask = self._srng.binomial(mask_shape, p=retain_prob,
                                       dtype=input.dtype)
            if self.shared_axes:
                bcast = tuple(bool(s == 1) for s in mask_shape)
                mask = T.patternbroadcast(mask, bcast)
            return input * mask

dropout = DropoutLayer  # shortcut


def dropout_channels(incoming, *args, **kwargs):
    pass

spatial_dropout = dropout_channels  # alias


def dropout_locations(incoming, *args, **kwargs):
    pass


class GaussianNoiseLayer(Layer):
    def __init__(self, incoming, sigma=0.1, **kwargs):
        super(GaussianNoiseLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.sigma = sigma

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true noise is disabled, see notes
        """
        if deterministic or self.sigma == 0:
            return input
        else:
            return input + self._srng.normal(input.shape,
                                             avg=0.0,
                                             std=self.sigma)
