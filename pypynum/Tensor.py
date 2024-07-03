from .Array import Array, fill
from .errors import ShapeError

MatchError = ShapeError("The shapes of two tensors do not match")


class Tensor(Array):
    """
    It is a tensor class that supports basic operations.
    :param data: An array in the form of a list
    :param check: Check the rationality of the input array
    """

    def __init__(self, data, check=True):
        super().__init__(data, check)

    def __add__(self, other):
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise MatchError
            return Tensor(fill(self.shape, [t1 + t2 for t1, t2 in zip(self.flatten(), other.flatten())], rtype=list),
                          False)
        elif isinstance(other, (int, float, complex)):
            return Tensor(tensor_and_number(self, "+", other), False)
        else:
            raise ValueError("The other must be a tensor or a number")

    def __sub__(self, other):
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise MatchError
            return Tensor(fill(self.shape, [t1 - t2 for t1, t2 in zip(self.flatten(), other.flatten())], rtype=list),
                          False)
        elif isinstance(other, (int, float, complex)):
            return Tensor(tensor_and_number(self, "-", other), False)
        else:
            raise ValueError("The other must be a tensor or a number")

    def __mul__(self, other):
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise MatchError
            return Tensor(fill(self.shape, [t1 * t2 for t1, t2 in zip(self.flatten(), other.flatten())], rtype=list),
                          False)
        elif isinstance(other, (int, float, complex)):
            return Tensor(tensor_and_number(self, "*", other), False)
        else:
            raise ValueError("The other must be a tensor or a number")

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            if self.shape[:-2] != other.shape[:-2]:
                raise MatchError

            def matmul(s, o):
                if not isinstance(s[0][0], list) or not isinstance(o[0][0], list):
                    return [[sum([s[i][k] * o[k][j] for k in range(len(s[0]))])
                             for j in range(len(o[0]))] for i in range(len(s))]
                if len(s) != len(o):
                    raise ValueError("Tensor dimensions do not match")
                return [matmul(t1, t2) for t1, t2 in zip(s, o)]

            return Tensor(matmul(self.data, other.data), False)
        else:
            raise ValueError("The other must be a tensor")


def tensor_and_number(tensor, operator, number):
    if isinstance(tensor, Tensor) or isinstance(tensor, list):
        _result = []
        for item in tensor:
            _result.append(tensor_and_number(item, operator, number))
        return _result
    else:
        if operator in ["+", "-", "*"]:
            return eval("{} {} {}".format(tensor, operator, number))


def zeros(_dimensions):
    from .Array import zeros as _zeros
    return Tensor(_zeros(_dimensions))


def zeros_like(_nested_list):
    from .Array import zeros_like as _zeros_like
    return Tensor(_zeros_like(_nested_list.data))


def ten(data: list) -> Tensor:
    return Tensor(data)


def tensorproduct(*tensors: Tensor) -> Tensor:
    if not all([isinstance(tensor, Tensor) for tensor in tensors]):
        raise TypeError("All inputs must be tensors")
    if len(tensors) == 1:
        return tensors[0].copy()

    def mul(a, b):
        return Tensor(fill(a.shape + b.shape, [i * j for i in a.flatten() for j in b.flatten()], rtype=list), False)

    first = tensors[0]
    for second in tensors[1:]:
        first = mul(first, second)
    return first


del Array
