import warnings

import theano

from .. import init
from .. import nonlinearities

from .base import Layer

from .conv import conv_output_length, BaseConvLayer
from ..utils import as_tuple

try:
    from theano import gpuarray as gpu
except ImportError:
    from theano.sandbox import gpuarray as gpu
gpu_enabled = gpu.pygpu_activated
if not gpu_enabled:
    try:
        from theano.sandbox import cuda as gpu
    except Exception:  # Theano 0.10+ raises nose.SkipTest
        gpu_enabled = False
    else:
        gpu_enabled = gpu.cuda_enabled
if gpu_enabled:
    gpu_contiguous = gpu.basic_ops.gpu_contiguous
    GpuCorrMM = gpu.blas.GpuCorrMM
else:
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

__all__ = [
    "Conv2DMMLayer",
]


class Conv2DMMLayer(BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 flip_filters=False, num_groups=1, **kwargs):
        super(Conv2DMMLayer, self).__init__(incoming, num_filters, filter_size,
                                            stride, pad, untie_biases, W, b,
                                            nonlinearity, flip_filters,
                                            num_groups, n=2, **kwargs)
        border_mode = 'half' if self.pad == 'same' else self.pad
        extra_kwargs = {'num_groups': num_groups} if num_groups > 1 else {}
        self.corr_mm_op = GpuCorrMM(subsample=self.stride,
                                    border_mode=border_mode,
                                    **extra_kwargs)

    def convolve(self, input, **kwargs):
        filters = self.W
        if self.flip_filters:
            filters = filters[:, :, ::-1, ::-1]  # flip top-down, left-right

        contiguous_filters = gpu_contiguous(filters)
        contiguous_input = gpu_contiguous(input)
        conved = self.corr_mm_op(contiguous_input, contiguous_filters)
        return conved
