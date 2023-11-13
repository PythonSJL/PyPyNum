real = int | float


class Quaternion:
    def __init__(self, w: real, x: real, y: real, z: real):
        self.__quaternion = w, x, y, z
        self.__w, self.__x, self.__y, self.__z = self.__quaternion

    def __repr__(self):
        return "({}+{}i+{}j+{}k)".format(self.__w, self.__x, self.__y, self.__z)

    def __add__(self, other: "Quaternion"):
        return Quaternion(*list(map(lambda i, j: i + j, self.__quaternion, other.__quaternion)))

    def __sub__(self, other: "Quaternion"):
        return Quaternion(*list(map(lambda i, j: i - j, self.__quaternion, other.__quaternion)))

    def __mul__(self, other: "int | float | Quaternion"):
        if isinstance(other, Quaternion):
            w = self.__w * other.__w - self.__x * other.__x - self.__y * other.__y - self.__z * other.__z
            x = self.__w * other.__x + self.__x * other.__w + self.__y * other.__z - self.__z * other.__y
            y = self.__w * other.__y + self.__y * other.__w + self.__z * other.__x - self.__x * other.__z
            z = self.__w * other.__z + self.__z * other.__w + self.__x * other.__y - self.__y * other.__x
            return Quaternion(w, x, y, z)
        elif isinstance(other, (int, float)):
            return Quaternion(*[other * i for i in self.__quaternion])
        else:
            raise TypeError("Operation undefined.")

    def __rmul__(self, other: real):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        else:
            raise TypeError("Operation undefined.")

    def __truediv__(self, other: "int | float | Quaternion"):
        if isinstance(other, Quaternion):
            divisor = sum([i ** 2 for i in other.__quaternion])
            w = self.__w / other.__w
            x = (self.__x * other.__w - self.__y * other.__z + self.__z * other.__y) / divisor
            y = (self.__y * other.__w + self.__x * other.__z - self.__z * other.__x) / divisor
            z = (self.__z * other.__w - self.__y * other.__x + self.__x * other.__y) / divisor
            return Quaternion(w, x, y, z)
        elif isinstance(other, (int, float)):
            return Quaternion(*[i / other for i in self.__quaternion])
        else:
            raise TypeError("Operation undefined.")

    def __round__(self, n: int = None):
        if n is None:
            return Quaternion(*[round(i) for i in self.__quaternion])
        return Quaternion(*[round(i, n) for i in self.__quaternion])

    def __abs__(self):
        return self.norm()

    def __eq__(self, other: "Quaternion"):
        return all(list(map(lambda i, j: abs(i) == abs(j), self.__quaternion, other.__quaternion)))

    def __iadd__(self, other: "Quaternion"):
        return self + other

    def __isub__(self, other: "Quaternion"):
        return self - other

    def __imul__(self, other: "Quaternion"):
        return self * other

    def __itruediv__(self, other: "Quaternion"):
        return self / other

    def __setitem__(self, key: int, value: real):
        quaternion = list(self.__quaternion)
        quaternion[key] = value
        self.__quaternion = tuple(quaternion)
        self.__w, self.__x, self.__y, self.__z = self.__quaternion

    def __pos__(self):
        return Quaternion(*self.__quaternion)

    def __neg__(self):
        return Quaternion(*[-i for i in self.__quaternion])

    def data(self):
        return self.__quaternion

    def norm(self):
        return sum([i ** 2 for i in self.__quaternion]) ** 0.5

    def conjugate(self):
        return Quaternion(self.__w, -self.__x, -self.__y, -self.__z)

    def normalize(self):
        norm = self.norm()
        return Quaternion(*[i / norm for i in self.__quaternion])

    def inverse(self):
        norm = self.norm()
        return Quaternion(*[i / norm for i in self.conjugate().__quaternion])


def init(number: real = 0):
    return Quaternion(number, number, number, number)


def quat(w: real, x: real, y: real, z: real):
    return Quaternion(w, x, y, z)
