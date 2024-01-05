from .errors import ShapeError

ArrayError = ShapeError("The shape of the array is invalid")


class Array:
    def __init__(self, data=None, check=True):
        def get_shape():
            _shape = []
            _sub = data
            while isinstance(_sub, list):
                _shape.append(len(_sub))
                _sub = _sub[0]
            return _shape

        if data is None:
            data = []
        self.shape = [] if data == [] else get_shape()
        if check and self.shape and not isinstance(data, (int, float, complex)):
            is_valid_array(data, self.shape)
        self.data = data

    def __repr__(self):
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

    def __getitem__(self, item):
        return self.data[item]

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)


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


def zeros(_dimensions):
    if len(_dimensions) == 0:
        return 0
    else:
        _array = []
        for i in range(_dimensions[0]):
            _row = zeros(_dimensions[1:])
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
