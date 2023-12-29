arr = list | tuple
num = int | float | complex


class Vector:
    def __init__(self, data):
        if not isinstance(data, arr) or not all([isinstance(item, num) for item in data]):
            raise TypeError("Each value of a vector must be a numerical value")
        self.len = len(data)
        self.data = data

    def __add__(self, other):
        if self.len != other.len:
            raise ValueError("Vector dimensions do not match")
        return Vector([self.data[i] + other.data[i] for i in range(self.len)])

    def __sub__(self, other):
        if self.len != other.len:
            raise ValueError("Vector dimensions do not match")
        return Vector([self.data[i] - other.data[i] for i in range(self.len)])

    def __mul__(self, other):
        if isinstance(other, Vector):
            if self.len != other.len:
                raise ValueError("Vector dimensions do not match")
            return Vector([self.data[i] * other.data[i] for i in range(self.len)])
        else:
            return Vector([i * other for i in self.data])

    def __matmul__(self, other):
        if self.len != other.len:
            raise ValueError("Vector dimensions do not match")
        return sum([self.data[i] * other.data[i] for i in range(self.len)])

    def __abs__(self):
        return self.norm()

    def __pos__(self):
        return Vector([abs(item) for item in self.data])

    def __neg__(self):
        return Vector([-item for item in self.data])

    def __round__(self, n=None):
        if n is None:
            return Vector([round(a) for a in self.data])
        return Vector([round(a, n) for a in self.data])

    def __eq__(self, other):
        return self.len == other.len and all([self.data[i] == other.data[i] for i in range(self.len)])

    def __ne__(self, other):
        return not self == other

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return str(self.data).replace("], ", "]\n").replace(",", "")

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
        elif isinstance(axes, int) and abs(axes) == axes:
            axis = Vector([1 if p == axes else 0 for p in range(self.len)])
            result = acos(self @ axis / (self.norm() * axis.norm()))
        elif isinstance(axes, arr):
            result = []
            for a in axes:
                axis = Vector([1 if p == a else 0 for p in range(self.len)])
                result.append(acos(self @ axis / (self.norm() * axis.norm())))
        else:
            raise ValueError("The axes parameter can only be None, natural number, list, or tuple")
        return result


def vec(data):
    return Vector(data)


def same(length, value=0):
    return Vector([value] * length)


# TODO 更多向量功能
...
