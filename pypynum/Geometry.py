import math

arr = list | tuple
real = int | float


class Line:
    def __init__(self, a: arr, b: arr):
        if isinstance(a, arr) and isinstance(b, arr):
            if len(a) == len(b) == 2:
                self.a, self.b = a, b
            else:
                raise ValueError("The coordinate value length can only be two")
        else:
            raise TypeError("The type of coordinate value can only be an array")

    def length(self) -> float:
        return ((self.a[0] - self.b[0]) ** 2 + (self.a[1] - self.b[1]) ** 2) ** 0.5

    def expr(self) -> list:
        if self.b[0] - self.a[0] == 0:
            k = None
        else:
            k = (self.b[1] - self.a[1]) / (self.b[0] - self.a[0])
        if k is None:
            b = None
        else:
            b = self.a[1] - k * self.a[0]
        return [k, b]

    def vertical(self, p: arr) -> tuple:
        if isinstance(p, arr):
            if len(p) == 2:
                k, b = self.expr()
                if k == 0:
                    x = p[0]
                    y = b
                    return x, y
                if k is None:
                    x = self.a[0]
                    y = p[1]
                    return x, y
                kp = -1 / k
                bp = p[1] - kp * p[0]
                x = (bp - b) / (k - kp)
                y = kp * x + bp
                return x, y
            else:
                raise ValueError("The coordinate value length can only be two")
        else:
            raise TypeError("The type of coordinate value can only be an array")


class Triangle:
    def __init__(self, a: arr, b: arr, c: arr):
        if isinstance(a, arr) and isinstance(b, arr) and isinstance(c, arr):
            if len(a) == len(b) == len(c) == 2:
                self.a, self.b, self.c = a, b, c
            else:
                raise ValueError("The coordinate value length can only be two")
        else:
            raise TypeError("The type of coordinate value can only be an array")

    def length(self) -> tuple:
        def distance(p1: arr, p2: arr) -> float:
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        return distance(self.a, self.b), distance(self.b, self.c), distance(self.c, self.a)

    def perimeter(self) -> float:
        return sum(self.length())

    def area(self) -> float:
        return abs(
            (self.b[0] - self.a[0]) * (self.c[1] - self.a[1]) - (self.b[1] - self.a[1]) * (self.c[0] - self.a[0])) / 2

    def incenter(self) -> tuple:
        a, b, c = self.length()
        x = (a * self.a[0] + b * self.b[0] + c * self.c[0]) / (a + b + c)
        y = (a * self.a[1] + b * self.b[1] + c * self.c[1]) / (a + b + c)
        return x, y

    def circumcenter(self) -> tuple:
        x = (self.a[0] * self.b[0] + self.b[0] * self.c[0] + self.c[0] * self.a[0]) / (
                self.a[0] + self.b[0] + self.c[0])
        y = (self.a[1] * self.b[1] + self.b[1] * self.c[1] + self.c[1] * self.a[1]) / (
                self.a[1] + self.b[1] + self.c[1])
        return x, y

    def centroid(self) -> tuple:
        x = (self.a[0] + self.b[0] + self.c[0]) / 3
        y = (self.a[1] + self.b[1] + self.c[1]) / 3
        return x, y

    def orthocenter(self) -> tuple:
        if self.area() == 0:
            raise ValueError("A triangle with zero area has no orthogonal center")
        a, b, c = self.a, self.b, self.c
        divisor = a[0] * b[1] - a[0] * c[1] - a[1] * b[0] + a[1] * c[0] + b[0] * c[1] - b[1] * c[0]
        x = (-a[0] * a[1] * b[0] + a[0] * a[1] * c[0] + a[0] * b[0] * b[1] - a[0] * c[0] * c[1] - a[1] ** 2 * b[1] + a[
            1] ** 2 * c[1] + a[1] * b[1] ** 2 - a[1] * c[1] ** 2 - b[0] * b[1] * c[0] + b[0] * c[0] * c[1] - b[1] ** 2 *
             c[1] + b[1] * c[1] ** 2) / divisor
        y = (a[0] ** 2 * b[0] - a[0] ** 2 * c[0] + a[0] * a[1] * b[1] - a[0] * a[1] * c[1] - a[0] * b[0] ** 2 + a[0] *
             c[0] ** 2 - a[1] * b[0] * b[1] + a[1] * c[0] * c[1] + b[0] ** 2 * c[0] + b[0] * b[1] * c[1] - b[0] * c[
                 0] ** 2 - b[1] * c[0] * c[1]) / divisor
        return x, y

    def is_isosceles(self, error: real = 0) -> bool:
        _length = self.length()
        return True if abs(_length[0] - _length[1]) <= error or abs(_length[1] - _length[2]) <= error or abs(
            _length[2] - _length[0]) <= error else False

    def is_equilateral(self, error: real = 0) -> bool:
        _length = self.length()
        return True if abs(_length[0] - _length[1]) <= error and abs(_length[1] - _length[2]) <= error and abs(
            _length[2] - _length[0]) <= error else False

    def is_right(self, error: real = 0) -> bool:
        _length = sorted(self.length())
        return True if abs(
            _length[0] * _length[0] + _length[1] * _length[1] - _length[2] * _length[2]) <= error else False


class Quadrilateral:
    def __init__(self, a: arr, b: arr, c: arr, d: arr):
        if isinstance(a, arr) and isinstance(b, arr) and isinstance(c, arr) and isinstance(d, arr):
            if len(a) == len(b) == len(c) == len(d) == 2:
                self.a, self.b, self.c, self.d = a, b, c, d
            else:
                raise ValueError("The coordinate value length can only be two")
        else:
            raise TypeError("The type of coordinate value can only be an array")

    def length(self) -> tuple:
        def distance(p1: arr, p2: arr) -> float:
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        return distance(self.a, self.b), distance(self.b, self.c), distance(self.c, self.d), distance(self.d, self.a)

    def perimeter(self) -> float:
        return sum(self.length())

    def area(self) -> float:
        return abs(
            (self.b[0] - self.a[0]) * (self.c[1] - self.a[1]) - (self.b[1] - self.a[1]) * (self.c[0] - self.a[0]) - (
                    self.d[0] - self.a[0]) * (self.c[1] - self.a[1]) + (self.d[1] - self.a[1]) * (
                    self.c[0] - self.a[0])) / 2

    def centroid(self) -> tuple:
        _ = Triangle(self.a, self.b, self.c)
        x1, y1 = _.centroid()
        s1 = _.area()
        _ = Triangle(self.a, self.d, self.c)
        x2, y2 = _.centroid()
        s2 = _.area()
        x = (x1 * s1 + x2 * s2) / (s1 + s2)
        y = (y1 * s1 + y2 * s2) / (s1 + s2)
        return x, y

    def _(self):
        return [Line(self.a, self.b).expr()[0], Line(self.c, self.d).expr()[0], Line(self.b, self.c).expr()[0],
                Line(self.d, self.a).expr()[0]]

    def is_trapezoidal(self, error: real = 0) -> bool:
        slopes = self._()
        slopes = [_ if _ is not None else 0 for _ in slopes]
        return True if abs(slopes[0] - slopes[1]) <= error or abs(slopes[2] - slopes[3]) <= error else False

    def is_parallelogram(self, error: real = 0) -> bool:
        slopes = self._()
        slopes = [_ if _ is not None else 0 for _ in slopes]
        return True if abs(slopes[0] - slopes[1]) <= error and abs(slopes[2] - slopes[3]) <= error else False

    def is_diamond(self, error: real = 0) -> bool:
        _length = self.length()
        return True if abs(_length[0] - _length[1]) <= error and abs(_length[1] - _length[2]) <= error and abs(
            _length[2] - _length[3]) <= error and abs(_length[3] - _length[0]) <= error else False

    def is_rectangular(self, error: real = 0) -> bool:
        slopes = [Line(self.a, self.b).expr()[0], Line(self.b, self.c).expr()[0]]
        if None in slopes:
            slopes.remove(None)
            if slopes:
                if abs(slopes[0]) <= error:
                    return True
                return False
            return False
        return True if abs(1 / max(slopes) + min(slopes)) <= error and self.is_parallelogram(error) else False

    def is_square(self, error: real = 0) -> bool:
        return True if self.is_diamond(error) and self.is_rectangular(error) else False


class Polygon:
    def __init__(self, *p: arr):
        if len(p) < 3:
            raise ValueError("The number of vertices in a polygon cannot be less than three")
        for item in p:
            if isinstance(item, arr):
                if len(item) != 2:
                    raise ValueError("The coordinate value length can only be two")
            else:
                raise TypeError("The type of coordinate value can only be an array")
        self.points = p

    def length(self) -> tuple:
        def distance(p1: arr, p2: arr) -> float:
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        return tuple(distance(*item) for item in zip(self.points, self.points[1:] + (self.points[0],)))

    def perimeter(self) -> float:
        return sum(self.length())

    def area(self) -> float:
        _area = 0
        q = self.points[-1]
        for p in self.points:
            _area += p[0] * q[1] - p[1] * q[0]
            q = p
        return abs(_area) / 2

    def centroid(self) -> tuple:
        raise NotImplementedError


class Circle:
    def __init__(self, center: arr, radius: real):
        if isinstance(center, arr):
            if len(center) == 2:
                self.center = center
            else:
                raise ValueError("The coordinate value length can only be two")
        else:
            raise TypeError("The type of coordinate value can only be an array")
        if isinstance(radius, real):
            self.radius = radius
        else:
            raise TypeError("The type of length value can only be real numbers")

    def perimeter(self) -> float:
        return 6.283185307179586 * self.radius

    def area(self) -> float:
        return 3.141592653589793 * self.radius * self.radius

    def expr(self) -> list:
        return [self.center[0], self.center[1], self.radius]

    def chord(self, radian: real) -> float:
        return 2 * self.radius * math.sin(radian / 2)
