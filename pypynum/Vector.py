from .Array import Array
from .errors import ShapeError

MatchError = ShapeError("The shapes of two vectors do not match")


class Vector(Array):
    """
    It is a vector class that supports basic operations,
    as well as functions such as norm and all angles.
    :param data: An array in the form of a list
    :param check: Check the rationality of the input array
    """

    def __init__(self, data, check=True):
        super().__init__(data, check)
        if len(self.shape) != 1:
            raise ShapeError("Vectors can only be one-dimensional in shape")
        self.len = len(data)

    def __add__(self, other):
        if self.len != other.len:
            raise MatchError
        return Vector([self.data[i] + other.data[i] for i in range(self.len)])

    def __sub__(self, other):
        if self.len != other.len:
            raise MatchError
        return Vector([self.data[i] - other.data[i] for i in range(self.len)])

    def __mul__(self, other):
        if isinstance(other, Vector):
            if self.len != other.len:
                raise MatchError
            return Vector([self.data[i] * other.data[i] for i in range(self.len)])
        else:
            return Vector([i * other for i in self.data])

    def __matmul__(self, other):
        if self.len != other.len:
            raise MatchError
        return sum([self.data[i] * other.data[i] for i in range(self.len)])

    def __abs__(self):
        return self.norm()

    def __pos__(self):
        return Vector([item for item in self.data])

    def __neg__(self):
        return Vector([-item for item in self.data])

    def norm(self, p=2):
        if p == 0:
            return len([item for item in self.data if item != 0])
        elif p == 1:
            return sum([abs(item) for item in self.data])
        elif p == 2:
            return sum([i ** 2 for i in self.data]) ** 0.5
        elif p == float("inf"):
            return max([abs(item) for item in self.data])
        else:
            return sum([i ** p for i in self.data]) ** (1 / p)

    def normalize(self):
        norm = self.norm()
        return Vector([i / norm for i in self.data])

    def angles(self, axes=None):
        from math import acos
        if axes is None or axes == [] or axes == ():
            result = []
            for a in range(self.len):
                axis = Vector([1 if p == a else 0 for p in range(self.len)])
                result.append(acos(self @ axis / (self.norm() * axis.norm())))
        elif int(abs(axes)) == axes:
            axis = Vector([1 if p == axes else 0 for p in range(self.len)])
            result = acos(self @ axis / (self.norm() * axis.norm()))
        elif isinstance(axes, (list, tuple)):
            result = []
            for a in axes:
                axis = Vector([1 if p == a else 0 for p in range(self.len)])
                result.append(acos(self @ axis / (self.norm() * axis.norm())))
        else:
            raise TypeError("The axes parameter can only be None, natural number, list, or tuple")
        return result


def vec(data):
    return Vector(data)


del Array
