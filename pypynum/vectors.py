from .arrays import Array
from .types import ShapeError

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

    def __matmul__(self, other):
        if self.len != other.len:
            raise MatchError
        return sum([x * y for x, y in zip(self.data, other.data)])

    def __abs__(self):
        return self.norm()

    def __pos__(self):
        return Vector([x for x in self.data])

    def __neg__(self):
        return Vector([-x for x in self.data])

    def norm(self, p=2):
        if p == 0:
            return len([x for x in self.data if x != 0])
        elif p == 1:
            return sum(map(abs, self.data))
        elif p == 2:
            return sum([x ** 2 for x in self.data]) ** 0.5
        elif p == float("inf"):
            return max(map(abs, self.data))
        else:
            return sum([x ** p for x in self.data]) ** (1 / p)

    def normalize(self):
        norm = self.norm()
        return Vector([x / norm for x in self.data])

    def cosine(self, other):
        if self.len != other.len:
            raise MatchError
        dot_product = self @ other
        norm_self = self.norm()
        norm_other = other.norm()
        return dot_product / (norm_self * norm_other)

    def angles(self, axes=None):
        from math import acos
        identity = []
        for i in range(self.len):
            unit_vector = [0] * self.len
            unit_vector[i] = 1
            identity.append(Vector(unit_vector))
        if axes is None or axes == [] or axes == ():
            result = [acos(self @ axis / (self.norm() * axis.norm())) for axis in identity]
        elif isinstance(axes, int) and axes >= 0:
            axis = identity[axes]
            result = acos(self @ axis / (self.norm() * axis.norm()))
        elif isinstance(axes, (list, tuple)):
            result = [acos(self @ identity[a] / (self.norm() * identity[a].norm())) for a in axes]
        else:
            raise TypeError("The axes parameter can only be None, non-negative integer, list, or tuple")
        return result

    def chebyshev(self, other):
        if self.len != other.len:
            raise MatchError
        return max([abs(x - y) for x, y in zip(self.data, other.data)])

    def manhattan(self, other):
        if self.len != other.len:
            raise MatchError
        return sum([abs(x - y) for x, y in zip(self.data, other.data)])

    def euclidean(self, other):
        if self.len != other.len:
            raise MatchError
        return sum([(x - y) ** 2 for x, y in zip(self.data, other.data)]) ** 0.5

    def minkowski(self, other, p):
        if self.len != other.len:
            raise MatchError
        return sum([abs(x - y) ** p for x, y in zip(self.data, other.data)]) ** (1 / p)


def vec(data):
    return Vector(data)


del Array
