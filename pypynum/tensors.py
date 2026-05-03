from .arrays import Array, fill
from .types import ShapeError

MatchError = ShapeError("The shapes of two tensors do not match")


class Tensor(Array):
    """
    It is a tensor class that supports basic operations.
    :param data: An array in the form of a list
    :param check: Check the rationality of the input array
    """

    def __init__(self, data, check=True):
        from warnings import warn
        warn("The 'Tensor' class is deprecated and will be removed in a future version. "
             "Please use the 'Array' class instead for similar functionality.", FutureWarning)
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


def ten(data: list) -> Tensor:
    return Tensor(data)


del Array
