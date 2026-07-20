
import numpy as np
import theano.tensor as T

from ..utils import int_types



def conv1d_sc(input, filters, image_shape=None, filter_shape=None,
              border_mode='valid', subsample=(1,), filter_flip=True):
    pass


def conv1d_mc0(input, filters, image_shape=None, filter_shape=None,
               border_mode='valid', subsample=(1,), filter_flip=True,
               num_groups=1):
    pass


def conv1d_mc1(input, filters, image_shape=None, filter_shape=None,
               border_mode='valid', subsample=(1,), filter_flip=True,
               num_groups=1):
    pass


def conv1d_unstrided(input, filters, image_shape, filter_shape,
                     border_mode='valid', subsample=(1,), filter_flip=True,
                     implementation=conv1d_sc):
    pass


def conv1d_sd(input, filters, image_shape, filter_shape, border_mode='valid',
              subsample=(1,), filter_flip=True):
    pass


def conv1d_md(input, filters, image_shape, filter_shape, border_mode='valid',
              subsample=(1,), filter_flip=True):
    pass




