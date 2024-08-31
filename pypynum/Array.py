from .errors import ShapeError

ArrayError = ShapeError("The shape of the array is invalid")
MatchError = ShapeError("The shapes of the two arrays do not match")


class Array:
    """
    It is the base class of vectors, matrices, and tensors, supporting operations and many statistical functions.
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
        return "{}({})".format(self.__class__.__name__, self.data)

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

    def __getitem__(self, item):
        def get_item_recursive(result, indices):
            if indices == ():
                return result
            index = indices[0]
            if isinstance(index, int):
                return get_item_recursive(result[index], indices[1:])
            elif isinstance(index, (slice, list, tuple, range)):
                if isinstance(index, slice):
                    start = index.start
                    stop = index.stop
                    step = index.step
                    if step is None or step > 0:
                        start = 0 if start is None else start
                        stop = len(result) if stop is None else stop
                    else:
                        start = len(result) - 1 if start is None else start
                        stop = -1 if stop is None else stop
                    try:
                        index = range(start, stop, step if step is not None else 1)
                    except TypeError:
                        raise TypeError("Slice indices must be integers or None or have an __index__ method")
                return [get_item_recursive(result[i], indices[1:]) for i in index]
            else:
                raise TypeError("Valid indices are integers, slices (`:`), lists, tuples, ranges, and BoolArray")

        if isinstance(item, (int, slice)):
            return self.data[item]
        if isinstance(item, tuple):
            if len(item) > len(self.shape):
                raise IndexError("Too many indices for array: array is {}-dimensional, but {} were indexed".format(
                    len(self.shape), len(item)))
            else:
                return get_item_recursive(self.data, item)
        if isinstance(item, list):
            if all([isinstance(idx, int) for idx in item]):
                try:
                    return [self.data[idx] for idx in item]
                except IndexError:
                    raise IndexError("Valid indices are from -{} to {}".format(len(self.data), len(self.data) - 1))
        if isinstance(item, BoolArray):
            if self.shape != item.shape:
                raise MatchError
            return [value for value, flag in zip(self.flatten(), item.flatten()) if flag]
        raise TypeError("Valid indices are integers, slices (`:`), lists, tuples, ranges, and BoolArray")

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __round__(self, n=None):
        from .ufuncs import base_ufunc
        return base_ufunc(self, func=lambda number, digits: round(number.real, digits) + round(
            number.imag, digits) * 1j if isinstance(number, complex) else round(number, digits), args=[n])

    def __hash__(self):
        return hash(repr(self.data))

    def __truediv__(self, other):
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise MatchError
            return type(self)(fill(self.shape, [t1 / t2 for t1, t2 in zip(self.flatten(), other.flatten())]))
        elif isinstance(other, (int, float, complex)):
            from .ufuncs import divide
            return divide(self, other)
        else:
            raise ValueError("Another must be an array or number")

    def __floordiv__(self, other):
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise MatchError
            return type(self)(fill(self.shape, [t1 // t2 for t1, t2 in zip(self.flatten(), other.flatten())]))
        elif isinstance(other, (int, float, complex)):
            from .ufuncs import floor_divide
            return floor_divide(self, other)
        else:
            raise ValueError("Another must be an array or number")

    def __mod__(self, other):
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise MatchError
            return type(self)(fill(self.shape, [t1 % t2 for t1, t2 in zip(self.flatten(), other.flatten())]))
        elif isinstance(other, (int, float, complex)):
            from .ufuncs import modulo
            return modulo(self, other)
        else:
            raise ValueError("Another must be an array or number")

    def __pow__(self, _exp, _mod=None):
        if isinstance(_exp, (int, float, complex, Array)) and isinstance(
                _mod, (int, float, complex, Array, type(None))):
            from .ufuncs import power
            return power(self, _exp, _mod)
        else:
            raise ValueError("Exponential and modulus must both be arrays or numbers")

    def __gt__(self, other):
        return self.comparison(other, lambda x, y: x > y)

    def __ge__(self, other):
        return self.comparison(other, lambda x, y: x >= y)

    def __eq__(self, other):
        return self.comparison(other, lambda x, y: x == y)

    def __le__(self, other):
        return self.comparison(other, lambda x, y: x <= y)

    def __lt__(self, other):
        return self.comparison(other, lambda x, y: x < y)

    def __ne__(self, other):
        return self.comparison(other, lambda x, y: x != y)

    def comparison(self, other, func):
        from .ufuncs import apply, base_ufunc
        if isinstance(other, (int, float, complex)):
            return apply(self, lambda x: func(x, other), BoolArray)
        elif isinstance(other, Array):
            if self.shape != other.shape:
                raise ValueError("Shapes must be equal for element-wise comparison")
            return base_ufunc(self, other, func=func, rtype=BoolArray)
        else:
            raise TypeError("Unsupported operand type(s) for comparison: '{}' and '{}'".format(type(self), type(other)))

    def apply(self, func):
        from .ufuncs import apply
        return apply(self, func)

    def flatten(self):
        data = self.data
        while isinstance(data[0], list):
            data = sum(data, [])
        return data

    def reshape(self, shape, repeat=True, pad=0):
        return type(self)(fill(shape, self.flatten(), repeat, pad))

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
    return tuple(_shape)


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


def aslist(data):
    if isinstance(data, list):
        return data
    if isinstance(data, Array):
        return data.data
    if isinstance(data, (tuple, str)):
        return list(data)
    if isinstance(data, set):
        return sorted(data)
    raise TypeError("Unable to convert to array type")


def asarray(data):
    return Array(aslist(data))


def full(shape, fill_value, rtype=Array):
    def inner(data):
        return fill_value if len(data) == 0 else [(inner(data[1:])) for _ in range(data[0])]

    if isinstance(fill_value, list):
        raise TypeError("The filled value cannot be a list")
    result = inner(shape)
    return result if rtype is list else rtype(result)


def full_like(a, fill_value, rtype=Array):
    def inner(data):
        return [inner(item) for item in data] if isinstance(data, list) else fill_value

    if isinstance(fill_value, list):
        raise TypeError("The filled value cannot be a list")
    if isinstance(a, Array):
        a = a.data
    result = inner(a)
    return result if rtype is list else rtype(result)


def zeros(shape, rtype=Array):
    return full(shape, 0, rtype)


def zeros_like(a, rtype=Array):
    return full_like(a, 0, rtype)


def ones(shape, rtype=Array):
    return full(shape, 1, rtype)


def ones_like(a, rtype=Array):
    return full_like(a, 1, rtype)


def fill(shape, sequence=None, repeat=True, pad=0, rtype=Array):
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
            return [inner(_shape[1:]) for _ in range(_shape[0])]

    result = inner(shape)
    return result if rtype is list else rtype(result)


class BoolArray(Array):
    def __init__(self, data=None, check=True):
        from .ufuncs import apply
        super().__init__(apply(Array(data, check), bool, list), False)

    def logic_op(self, other, func):
        from .ufuncs import ufunc_helper
        if isinstance(other, Array) and not isinstance(other, BoolArray) and not isinstance(other, bool):
            raise TypeError("Other must be a BoolArray or a scalar boolean value")
        return ufunc_helper(self, other, func)

    def __and__(self, other):
        return self.logic_op(other, lambda a, b: a and b)

    def __or__(self, other):
        return self.logic_op(other, lambda a, b: a or b)

    def __xor__(self, other):
        return self.logic_op(other, lambda a, b: a != b)

    def __invert__(self):
        return self.apply(lambda x: not x)


def boolarray(data=None):
    return BoolArray(data)
