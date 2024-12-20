from .matrices import Matrix
from .types import ShapeError, Union, real


class Quaternion:
    def __init__(self, w: real, x: real, y: real, z: real):
        self.__quaternion = w, x, y, z
        self.__w, self.__x, self.__y, self.__z = self.__quaternion

    def __repr__(self):
        return "({}+{}i+{}j+{}k)".format(*self.__quaternion).replace("+-", "-")

    def __add__(self, other: "Quaternion"):
        return Quaternion(*tuple(map(lambda i, j: i + j, self.__quaternion, other.__quaternion)))

    def __sub__(self, other: "Quaternion"):
        return Quaternion(*tuple(map(lambda i, j: i - j, self.__quaternion, other.__quaternion)))

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
            return NotImplemented

    def __rmul__(self, other: real):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        else:
            return NotImplemented

    def __truediv__(self, other: "int | float | Quaternion"):
        if isinstance(other, Quaternion):
            return self * other.inverse()
        elif isinstance(other, (int, float)):
            return Quaternion(*[i / other for i in self.__quaternion])
        else:
            return NotImplemented

    def __round__(self, n: int = None):
        if n is None:
            return Quaternion(*[round(i) for i in self.__quaternion])
        return Quaternion(*[round(i, n) for i in self.__quaternion])

    def __abs__(self):
        return self.norm()

    def __eq__(self, other: "Quaternion"):
        return self.__quaternion == other.__quaternion

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
        return list(self.__quaternion)

    def norm(self):
        return sum([i ** 2 for i in self.__quaternion]) ** 0.5

    def conjugate(self):
        return Quaternion(self.__w, -self.__x, -self.__y, -self.__z)

    def normalize(self):
        norm = self.norm()
        return Quaternion(*[i / norm for i in self.__quaternion])

    def inverse(self):
        norm2 = sum([i ** 2 for i in self.__quaternion])
        if norm2 == 0:
            raise ZeroDivisionError("Cannot invert zero quaternion")
        return Quaternion(*[i / norm2 for i in self.conjugate().__quaternion])


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
        return list(self.__angles)


def euler(yaw: real = 0, pitch: real = 0, roll: real = 0) -> Euler:
    return Euler(yaw, pitch, roll)


def convert(data: Union[Quaternion, Matrix, Euler], to: str) -> Union[Quaternion, Matrix, Euler]:
    if to not in ["Q", "M", "E"]:
        raise ValueError("The parameter 'to' can only be 'Q', 'M', or 'E'")
    elif isinstance(data, Euler):
        if to == "Q":
            from math import cos, sin
            q = [x / 2.0 for x in data.data()]
            w = sin(q[0]) * sin(q[1]) * sin(q[2]) + cos(q[0]) * cos(q[1]) * cos(q[2])
            x = -cos(q[0]) * sin(q[1]) * sin(q[2]) + sin(q[0]) * cos(q[1]) * cos(q[2])
            y = sin(q[0]) * cos(q[1]) * sin(q[2]) + cos(q[0]) * sin(q[1]) * cos(q[2])
            z = -sin(q[0]) * sin(q[1]) * cos(q[2]) + cos(q[0]) * cos(q[1]) * sin(q[2])
            return Quaternion(w, x, y, z)
    elif isinstance(data, Matrix):
        if data.shape != [3, 3]:
            raise ShapeError("The rotation matrix must be a third-order square matrix")
        elif to == "Q":
            [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]] = data
            squares = [m11 + m22 + m33, m11 - m22 - m33, m22 - m11 - m33, m33 - m11 - m22]
            max_square_sum = max(squares)
            max_index = squares.index(max_square_sum)
            sqrt_val = (max_square_sum + 1.0) ** 0.5 * 0.5
            mult = 0.25 / sqrt_val
            if max_index == 0:
                w = max_index
                x = (m23 - m32) * mult
                y = (m31 - m13) * mult
                z = (m12 - m21) * mult
            elif max_index == 1:
                x = max_index
                w = (m23 - m32) * mult
                y = (m12 + m21) * mult
                z = (m31 + m13) * mult
            elif max_index == 2:
                y = max_index
                w = (m31 - m13) * mult
                x = (m12 + m21) * mult
                z = (m23 + m32) * mult
            else:
                z = max_index
                w = (m12 - m21) * mult
                x = (m31 + m13) * mult
                y = (m23 + m32) * mult
            return Quaternion(w, x, y, z)
    elif isinstance(data, Quaternion):
        if to == "M":
            w, x, y, z = data.normalize().data()
            return Matrix([[1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                           [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
                           [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]], False)
        elif to == "E":
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
        raise TypeError("Data can only be Euler or Matrix or Quaternion")


class Octonion:
    def __init__(self, s: real, t: real, u: real, v: real, w: real, x: real, y: real, z: real):
        self.__octonion = s, t, u, v, w, x, y, z
        self.__s, self.__t, self.__u, self.__v, self.__w, self.__x, self.__y, self.__z = self.__octonion

    def __repr__(self):
        return "({}+{}i+{}j+{}k+{}l+{}m+{}n+{}o)".format(*self.__octonion).replace("+-", "-")

    def __add__(self, other: "Octonion"):
        ss, st, su, sv, sw, sx, sy, sz = self.__octonion
        os, ot, ou, ov, ow, ox, oy, oz = other.__octonion
        return Octonion(ss + os, st + ot, su + ou, sv + ov, sw + ow, sx + ox, sy + oy, sz + oz)

    def __sub__(self, other: "Octonion"):
        ss, st, su, sv, sw, sx, sy, sz = self.__octonion
        os, ot, ou, ov, ow, ox, oy, oz = other.__octonion
        return Octonion(ss - os, st - ot, su - ou, sv - ov, sw - ow, sx - ox, sy - oy, sz - oz)

    def __mul__(self, other: "int | float | Octonion"):
        if isinstance(other, Octonion):
            ss, st, su, sv, sw, sx, sy, sz = self.__octonion
            os, ot, ou, ov, ow, ox, oy, oz = other.__octonion
            s = ss * os - st * ot - su * ou - sv * ov - sw * ow - sx * ox - sy * oy - sz * oz
            t = ss * ot + st * os + su * ov - sv * ou + sw * ox - sx * ow - sy * oz + sz * oy
            u = ss * ou - st * ov + su * os + sv * ot + sw * oy + sx * oz - sy * ow - sz * ox
            v = ss * ov + st * ou - su * ot + sv * os + sw * oz - sx * oy + sy * ox - sz * ow
            w = ss * ow - st * ox - su * oy - sv * oz + sw * os + sx * ot + sy * ou + sz * ov
            x = ss * ox + st * ow - su * oz + sv * oy - sw * ot + sx * os - sy * ov + sz * ou
            y = ss * oy + st * oz + su * ow - sv * ox - sw * ou + sx * ov + sy * os - sz * ot
            z = ss * oz - st * oy + su * ox + sv * ow - sw * ov - sx * ou + sy * ot + sz * os
            return Octonion(s, t, u, v, w, x, y, z)
        elif isinstance(other, (int, float)):
            ss, st, su, sv, sw, sx, sy, sz = self.__octonion
            return Octonion(ss * other, st * other, su * other, sv * other,
                            sw * other, sx * other, sy * other, sz * other)
        else:
            return NotImplemented

    def __rmul__(self, other: real):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        else:
            return NotImplemented

    def __truediv__(self, other: "int | float | Octonion"):
        if isinstance(other, Octonion):
            return self * other.inverse()
        elif isinstance(other, (int, float)):
            ss, st, su, sv, sw, sx, sy, sz = self.__octonion
            return Octonion(ss / other, st / other, su / other, sv / other,
                            sw / other, sx / other, sy / other, sz / other)
        else:
            return NotImplemented

    def __round__(self, n: int = None):
        if n is None:
            return Octonion(*[round(i) for i in self.__octonion])
        return Octonion(*[round(i, n) for i in self.__octonion])

    def __abs__(self):
        return self.norm()

    def __eq__(self, other: "Octonion"):
        return self.__octonion == other.__octonion

    def __iadd__(self, other: "Octonion"):
        return self + other

    def __isub__(self, other: "Octonion"):
        return self - other

    def __imul__(self, other: "Octonion"):
        return self * other

    def __itruediv__(self, other: "Octonion"):
        return self / other

    def __setitem__(self, key: int, value: real):
        self.__octonion = list(self.__octonion)
        self.__octonion[key] = value
        self.__s, self.__t, self.__u, self.__v, self.__w, self.__x, self.__y, self.__z = self.__octonion

    def __pos__(self):
        return Octonion(*self.__octonion)

    def __neg__(self):
        return Octonion(*[-i for i in self.__octonion])

    def data(self):
        return list(self.__octonion)

    def norm(self):
        return sum([i ** 2 for i in self.__octonion]) ** 0.5

    def conjugate(self):
        return Octonion(self.__s, -self.__t, -self.__u, -self.__v, -self.__w, -self.__x, -self.__y, -self.__z)

    def normalize(self):
        norm = self.norm()
        return Octonion(*[i / norm for i in self.__octonion])

    def inverse(self):
        norm2 = sum([i ** 2 for i in self.__octonion])
        if norm2 == 0:
            raise ZeroDivisionError("Cannot invert zero octonion")
        return Octonion(*[i / norm2 for i in self.conjugate().__octonion])


def octo(s: real = 0, t: real = 0, u: real = 0, v: real = 0,
         w: real = 0, x: real = 0, y: real = 0, z: real = 0) -> Octonion:
    return Octonion(s, t, u, v, w, x, y, z)
