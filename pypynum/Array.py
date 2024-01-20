from .errors import ShapeError

ArrayError = ShapeError("The shape of the array is invalid")


class Array:
    def __init__(self, data=None, check=True):
        if data is None:
            data = []
        self.shape = [] if data == [] else get_shape(data)
        if check and self.shape and not isinstance(data, (int, float, complex)):
            is_valid_array(data, self.shape)
        self.data = data

    def __repr__(self):
        _type = str(type(self))
        return _type[_type.rindex(".") + 1:-2] + "({})".format(self.data)

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
                _item = str(_nested_list)
                return " " * (_max_length - len(_item)) + _item

        _max = len(max(str(self.data).replace("[", "").replace("]", "").replace(",", "").split(), key=len))
        _then = str(_format(self.data, _max)).replace("], ", "]\n").replace(",", "").replace("'", "").split("\n")
        _max = max([_.count("[") for _ in _then])
        return "\n".join([(_max - _.count("[")) * " " + _ + "\n" * (_.count("]") - 1) for _ in _then]).strip()

    def __eq__(self, other):
        return self.data == other.data

    def __ne__(self, other):
        return not self.data == other.data

    def __getitem__(self, item):
        return self.data[item]

    def __round__(self, n=None):
        return function(self, round, [n])

    def __hash__(self):
        return hash(repr(self.data))

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
        if not all(isinstance(_, (int, float, complex, Array)) for _ in _array):
            raise TypeError("The value of the array must be a number")
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


def fill(shape, sequence=None):
    pointer = -1
    length = 1
    for item in shape:
        length *= item
    if sequence is None:
        sequence = list(range(length))

    def inner(_shape):
        nonlocal pointer
        if len(_shape) == 0:
            pointer += 1
            return sequence[pointer % len(sequence)]
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

    return type(_array)(inner(data))
