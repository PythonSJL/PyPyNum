from random import choice as _choice, gauss as _gauss, randint as _randint, random as _random, uniform as _uniform
from .Array import array
from .errors import RandomError
from .types import Union, arr, ite, real


def choice(seq: ite, shape: arr = None):
    if not isinstance(seq, (list, tuple, str)):
        raise TypeError("The parameter seq must be iterable")
    if shape is None:
        return _choice(seq)
    if not (isinstance(shape, (list, tuple)) and all([isinstance(item, int) and item and abs(
            item) == item for item in shape])):
        raise RandomError("The shape must be all positive integers")

    def inner(_dimensions):
        if len(_dimensions) == 0:
            return _choice(seq)
        else:
            _array = []
            for i in range(_dimensions[0]):
                _row = inner(_dimensions[1:])
                _array.append(_row)
            return _array

    return inner(shape)


def gauss(mu: real = 0, sigma: real = 1, shape: arr = None) -> Union[float, list]:
    if not (isinstance(mu, (int, float)) and isinstance(sigma, (int, float))):
        raise RandomError("The parameters mu and sigma must both be real numbers")
    if shape is None:
        return _gauss(mu, sigma)
    if not (isinstance(shape, (list, tuple)) and all([isinstance(item, int) and item and abs(
            item) == item for item in shape])):
        raise RandomError("The shape must be all positive integers")

    def inner(_dimensions):
        if len(_dimensions) == 0:
            return _gauss(mu, sigma)
        else:
            _array = []
            for i in range(_dimensions[0]):
                _row = inner(_dimensions[1:])
                _array.append(_row)
            return _array

    return inner(shape)


def gauss_error(original: arr, mu: real = 0, sigma: real = 1) -> list:
    if not (isinstance(mu, (int, float)) and isinstance(sigma, (int, float))):
        raise RandomError("The parameters mu and sigma must both be real numbers")
    array(original)

    def inner(_nested_list):
        if isinstance(_nested_list, list):
            _copy = []
            for item in _nested_list:
                _copy.append(inner(item))
            return _copy
        else:
            return _nested_list + _gauss(mu, sigma)

    return inner(original)


def randint(a: int, b: int, shape: arr = None) -> Union[int, list]:
    if not (isinstance(a, int) and isinstance(b, int)):
        raise RandomError("The range must be all integers")
    if shape is None:
        return _randint(a, b)
    if not (isinstance(shape, (list, tuple)) and all([isinstance(item, int) and item and abs(
            item) == item for item in shape])):
        raise RandomError("The shape must be all positive integers")

    def inner(_dimensions):
        if len(_dimensions) == 0:
            return _randint(a, b)
        else:
            _array = []
            for i in range(_dimensions[0]):
                _row = inner(_dimensions[1:])
                _array.append(_row)
            return _array

    return inner(shape)


def rand(shape: arr = None) -> Union[float, list]:
    if shape is None:
        return _random()
    if not (isinstance(shape, (list, tuple)) and all([isinstance(item, int) and item and abs(
            item) == item for item in shape])):
        raise RandomError("The shape must be all positive integers")

    def inner(_dimensions):
        if len(_dimensions) == 0:
            return _random()
        else:
            _array = []
            for i in range(_dimensions[0]):
                _row = inner(_dimensions[1:])
                _array.append(_row)
            return _array

    return inner(shape)


def uniform(a: real, b: real, shape: arr = None) -> Union[float, list]:
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise RandomError("The range must be all real numbers")
    if shape is None:
        return _uniform(a, b)
    if not (isinstance(shape, (list, tuple)) and all([isinstance(item, int) and item and abs(
            item) == item for item in shape])):
        raise RandomError("The shape must be all positive integers")

    def inner(_dimensions):
        if len(_dimensions) == 0:
            return _uniform(a, b)
        else:
            _array = []
            for i in range(_dimensions[0]):
                _row = inner(_dimensions[1:])
                _array.append(_row)
            return _array

    return inner(shape)
