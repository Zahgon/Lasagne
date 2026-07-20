
from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T
from . import utils

__all__ = [
    "sgd",
    "apply_momentum",
    "momentum",
    "apply_nesterov_momentum",
    "nesterov_momentum",
    "adagrad",
    "rmsprop",
    "adadelta",
    "adam",
    "adamax",
    "amsgrad",
    "norm_constraint",
    "total_norm_constraint"
]


def get_or_compute_grads(loss_or_grads, params):
    pass


def sgd(loss_or_grads, params, learning_rate):
    pass


def apply_momentum(updates, params=None, momentum=0.9):
    pass


def momentum(loss_or_grads, params, learning_rate, momentum=0.9):
    pass


def apply_nesterov_momentum(updates, params=None, momentum=0.9):
    pass


def nesterov_momentum(loss_or_grads, params, learning_rate, momentum=0.9):
    pass


def adagrad(loss_or_grads, params, learning_rate=1.0, epsilon=1e-6):
    pass


def rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    pass


def adadelta(loss_or_grads, params, learning_rate=1.0, rho=0.95, epsilon=1e-6):
    pass


def adam(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    pass


def adamax(loss_or_grads, params, learning_rate=0.002, beta1=0.9,
           beta2=0.999, epsilon=1e-8):
    pass


def amsgrad(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
            beta2=0.999, epsilon=1e-8):
    pass


def norm_constraint(tensor_var, max_norm, norm_axes=None, epsilon=1e-7):
    pass


def total_norm_constraint(tensor_vars, max_norm, epsilon=1e-7,
                          return_norm=False):
    pass
