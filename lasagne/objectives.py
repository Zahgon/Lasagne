
import theano.tensor

from .utils import as_theano_expression

__all__ = [
    "binary_crossentropy",
    "categorical_crossentropy",
    "squared_error",
    "aggregate",
    "binary_hinge_loss",
    "multiclass_hinge_loss",
    "huber_loss",
    "binary_accuracy",
    "categorical_accuracy"
]


def align_targets(predictions, targets):
    pass


def binary_crossentropy(predictions, targets):
    pass


def categorical_crossentropy(predictions, targets):
    pass


def squared_error(a, b):
    pass


def aggregate(loss, weights=None, mode='mean'):
    pass


def binary_hinge_loss(predictions, targets, delta=1, log_odds=None,
                      binary=True):
    pass


def multiclass_hinge_loss(predictions, targets, delta=1):
    pass


def huber_loss(predictions, targets, delta=1):
    pass


def binary_accuracy(predictions, targets, threshold=0.5):
    pass


def categorical_accuracy(predictions, targets, top_k=1):
    pass
