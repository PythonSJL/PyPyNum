real = int | float


class Quaternion:
    def __init__(self, w: real, x: real, y: real, z: real):
        self.__quaternion = w, x, y, z
        self.__w, self.__x, self.__y, self.__z = self.__quaternion

    def __repr__(self):
        return "({}+{}i+{}j+{}k)".format(*self.__quaternion)

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
            raise TypeError("Operation undefined")

    def __rmul__(self, other: real):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        else:
            raise TypeError("Operation undefined")

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
            raise TypeError("Operation undefined")

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
        q = list(self.__quaternion)
        q[key] = value
        self.__quaternion = tuple(q)
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


def quat(w: real = 0, x: real = 0, y: real = 0, z: real = 0) -> Quaternion:
    return Quaternion(w, x, y, z)


class Euler:
    def __init__(self, y: real, p: real, r: real):
        self.__angles = y, p, r
        self.__y, self.__p, self.__r = self.__angles

    def __repr__(self):
        return "ypr({},{},{})".format(*self.__angles)

    def __round__(self, n: int = None):
        if n is None:
            return Quaternion(*[round(i) for i in self.__angles])
        return Quaternion(*[round(i, n) for i in self.__angles])

    def __setitem__(self, key: int, value: real):
        e = list(self.__angles)
        e[key] = value
        self.__angles = tuple(e)
        self.__y, self.__p, self.__r = self.__angles

    def data(self):
        return self.__angles


def euler(yaw: real = 0, pitch: real = 0, roll: real = 0) -> Euler:
    return Euler(yaw, pitch, roll)


def change(data: Euler | Quaternion) -> Quaternion | Euler:
    if isinstance(data, Euler):
        from math import cos, sin
        q = [x / 2.0 for x in data.data()]
        w = sin(q[0]) * sin(q[1]) * sin(q[2]) + cos(q[0]) * cos(q[1]) * cos(q[2])
        x = -cos(q[0]) * sin(q[1]) * sin(q[2]) + sin(q[0]) * cos(q[1]) * cos(q[2])
        y = sin(q[0]) * cos(q[1]) * sin(q[2]) + cos(q[0]) * sin(q[1]) * cos(q[2])
        z = -sin(q[0]) * sin(q[1]) * cos(q[2]) + cos(q[0]) * cos(q[1]) * sin(q[2])
        return Quaternion(w, x, y, z)
    elif isinstance(data, Quaternion):
        from math import asin, atan2
        w, x, y, z = data.normalize().data()
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        yaw = atan2(t0, t1)
        t2 = 2.0 * (w * y - z * x)
        pitch = asin(t2)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        roll = atan2(t3, t4)
        return Euler(yaw, pitch, roll)
    else:
        raise TypeError("Data can only be Euler or Quaternion")
