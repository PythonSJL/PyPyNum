from random import choice as __choice, gauss as __gauss, randint as __randint, random as __random, uniform as __uniform
from .errors import RandomError
from .types import Union, arr, ite, real


def __validate_shape(shape):
    if not (isinstance(shape, (list, tuple)) and all([isinstance(item, int) and item > 0 for item in shape])):
        raise RandomError("The shape must be all positive integers")


def __create_nested_list(dimensions, func):
    return func() if len(dimensions) == 0 else [__create_nested_list(dimensions[1:], func)
                                                for _ in range(dimensions[0])]


def choice(seq: ite, shape: arr = None):
    if not isinstance(seq, (list, tuple, str)):
        raise TypeError("The parameter seq must be iterable")
    if shape is not None:
        __validate_shape(shape)
        return __create_nested_list(shape, lambda: __choice(seq))
    return __choice(seq)


def gauss(mu: real = 0, sigma: real = 1, shape: arr = None) -> Union[float, list]:
    if not (isinstance(mu, (int, float)) and isinstance(sigma, (int, float))):
        raise RandomError("The parameters mu and sigma must both be real numbers")
    if shape is not None:
        __validate_shape(shape)
        return __create_nested_list(shape, lambda: __gauss(mu, sigma))
    return __gauss(mu, sigma)


def randint(a: int, b: int, shape: arr = None) -> Union[int, list]:
    if not (isinstance(a, int) and isinstance(b, int)):
        raise RandomError("The range must be all integers")
    if shape is not None:
        __validate_shape(shape)
        return __create_nested_list(shape, lambda: __randint(a, b))
    return __randint(a, b)


def rand(shape: arr = None) -> Union[float, list]:
    if shape is not None:
        __validate_shape(shape)
        return __create_nested_list(shape, __random)
    return __random()


def uniform(a: real, b: real, shape: arr = None) -> Union[float, list]:
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise RandomError("The range must be all real numbers")
    if shape is not None:
        __validate_shape(shape)
        return __create_nested_list(shape, lambda: __uniform(a, b))
    return __uniform(a, b)
