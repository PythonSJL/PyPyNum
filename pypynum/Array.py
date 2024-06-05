from .errors import ShapeError
from .ufuncs import *

ArrayError = ShapeError("The shape of the array is invalid")
MatchError = ShapeError("The shapes of the two arrays do not match")


class Array:
    """
    It is the base class of vectors, matrices, and Arrays,
    which does not support operations but has many statistical functions.
    :param data: An array in the form of a list
    :param check: Check the rationality of the input array
    """

    def __init__(self, data=None, check=True):
        if data is None:
            data = []
        self.shape = [] if data == [] else get_shape(data)
        if check and self.shape and not isinstance(data, (int, float, complex)):
            is_valid_array(data, self.shape)
        self.data = data

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.data)

    def __str__(self):
        if not self.data:
            return "[]"

        def _format(_nested_list, _max_length):
            if isinstance(_nested_list, list):
                _copy = []
                for item in _nested_list:
                    _copy.append(_format(item, _max_length))
                return _copy
            else:
                _item = repr(_nested_list)
                return " " * (_max_length - len(_item)) + _item

        _max = len(max(str(self.data).replace("[", "").replace("]", "").replace(",", "").split(), key=len))
        _then = str(_format(self.data, _max)).replace("], ", "]\n").replace(",", "").replace("'", "").split("\n")
        _max = max([_.count("[") for _ in _then])
        return "\n".join([(_max - _.count("[")) * " " + _ + "\n" * (_.count("]") - 1) for _ in _then]).strip()

    def __add__(self, other):
        return Array(self.copy().data + other.copy().data)

    def __radd__(self, other):
        return self + other

    def __eq__(self, other):
        return self.data == other.data

    def __ne__(self, other):
        return self.data != other.data

    def __getitem__(self, item):
        return self.data[item]

    def __round__(self, n=None):
        return function(self, lambda number, digits: round(number.real, digits) + round(
            number.imag, digits) * 1j if isinstance(number, complex) else round(number, digits), [n])

    def __hash__(self):
        return hash(repr(self.data))

    def __truediv__(self, other):
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise MatchError
            return Array(fill(self.shape, [t1 / t2 for t1, t2 in zip(self.flatten(), other.flatten())]), False)
        elif isinstance(other, (int, float, complex)):
            return function(self, divide, [other])
        else:
            raise ValueError("Another must be an array or number")

    def __floordiv__(self, other):
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise MatchError
            return Array(fill(self.shape, [t1 // t2 for t1, t2 in zip(self.flatten(), other.flatten())]), False)
        elif isinstance(other, (int, float, complex)):
            return function(self, floor_divide, [other])
        else:
            raise ValueError("Another must be an array or number")

    def __mod__(self, other):
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise MatchError
            return Array(fill(self.shape, [t1 % t2 for t1, t2 in zip(self.flatten(), other.flatten())]), False)
        elif isinstance(other, (int, float, complex)):
            return function(self, modulo, [other])
        else:
            raise ValueError("Another must be an array or number")

    def __pow__(self, _exp, _mod=None):
        if isinstance(_exp, (int, float, complex)) and isinstance(_mod, (int, float, complex, type(None))):
            return function(self, power, [_exp, _mod])
        else:
            raise ValueError()

    def flatten(self):
        data = self.data
        while isinstance(data[0], list):
            data = sum(data, [])
        return data

    def reshape(self, shape):
        return type(self)(fill(shape, self.flatten()))

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def basic(self, func, axis=None):
        if axis is None:
            return func(self.flatten())

        def inner(a, flag):
            if flag:
                return [inner(_, flag - 1) for _ in a]
            elif all([isinstance(_, (int, float, complex)) for _ in a]):
                return func(a)
            else:
                return [inner(list(_), 0) for _ in zip(*a)]

        return type(self)(inner(self, axis))

    def sum(self, axis=None):
        return self.basic(sum, axis)

    def max(self, axis=None):
        return self.basic(max, axis)

    def min(self, axis=None):
        return self.basic(min, axis)

    def argmax(self, axis=None):
        return self.basic(lambda a: a.index(max(a)), axis)

    def argmin(self, axis=None):
        return self.basic(lambda a: a.index(min(a)), axis)

    def mean(self, axis=None):
        from .maths import mean
        return self.basic(mean, axis)

    def var(self, axis=None):
        from .maths import var
        return self.basic(var, axis)

    def std(self, axis=None):
        from .maths import std
        return self.basic(std, axis)

    def ptp(self, axis=None):
        from .maths import ptp
        return self.basic(ptp, axis)

    def median(self, axis=None):
        from .maths import median
        return self.basic(median, axis)

    def mode(self, axis=None):
        from .maths import mode
        return self.basic(mode, axis)

    def product(self, axis=None):
        from .maths import product
        return self.basic(product, axis)


def get_shape(data):
    _shape = []
    _sub = data
    while isinstance(_sub, list):
        _shape.append(len(_sub))
        _sub = _sub[0]
    return _shape


def is_valid_array(_array, _shape):
    if len(_shape) != 1 and not isinstance(_array, list):
        raise ArrayError
    if len(_shape) == 1:
        if not isinstance(_array, list) or len(_array) != _shape[0]:
            raise ArrayError
    elif len(_array) == _shape[0]:
        for _ in _array:
            is_valid_array(_, _shape[1:])
    else:
        raise ArrayError


def array(data=None):
    return Array(data)


def zeros(shape):
    if len(shape) == 0:
        return 0
    else:
        _array = []
        for i in range(shape[0]):
            _row = zeros(shape[1:])
            _array.append(_row)
        return _array


def zeros_like(_nested_list):
    if isinstance(_nested_list, list):
        _copy = []
        for item in _nested_list:
            _copy.append(zeros_like(item))
        return _copy
    else:
        return 0


def fill(shape, sequence=None, repeat=True, pad=0):
    pointer = -1
    length = 1
    for item in shape:
        length *= item
    if sequence is None:
        sequence = list(range(length))
    total = len(sequence)
    last = total - 1

    def inner(_shape):
        nonlocal pointer
        if len(_shape) == 0:
            if pointer == last and not repeat:
                return pad
            pointer += 1
            return sequence[pointer % total]
        else:
            _array = []
            for i in range(_shape[0]):
                _row = inner(_shape[1:])
                _array.append(_row)
            return _array

    return inner(shape)


def function(_array, _function, args=None):
    _type = str(type(_function))
    if not isinstance(_array, Array) or not (_type.startswith("<function ") or _type.startswith("<class ")):
        raise TypeError("The input parameter type is incorrect")
    data = _array.data

    def inner(_array):
        if isinstance(_array, list):
            _copy = []
            for item in _array:
                _copy.append(inner(item))
            return _copy
        else:
            return _function(_array) if args is None else _function(_array, *args)

    return type(_array)(inner(data), False)
