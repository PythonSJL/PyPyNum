from .types import arr, real


class Point:
    def __init__(self, p: arr):
        if isinstance(p, (list, tuple)):
            if len(p) == 2:
                self.p = p
            else:
                raise ValueError("The coordinate value length can only be two")
        else:
            raise TypeError("The type of coordinate value can only be an array")

    def __repr__(self):
        return "Point({})".format(self.p)


class Line:
    def __init__(self, a: arr, b: arr):
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) == len(b) == 2:
                self.a, self.b = a, b
            else:
                raise ValueError("The coordinate value length can only be two")
        else:
            raise TypeError("The type of coordinate value can only be an array")

    def __repr__(self) -> str:
        return "Line({}, {})".format(self.a, self.b)

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
        if isinstance(p, (list, tuple)):
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
            raise ValueError("The coordinate value length can only be two")
        raise TypeError("The type of coordinate value can only be an array")


class Triangle:
    def __init__(self, a: arr, b: arr, c: arr):
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and isinstance(c, (list, tuple)):
            if len(a) == len(b) == len(c) == 2:
                self.a, self.b, self.c = a, b, c
            else:
                raise ValueError("The coordinate value length can only be two")
        else:
            raise TypeError("The type of coordinate value can only be an array")

    def __repr__(self) -> str:
        return "Triangle({}, {}, {})".format(self.a, self.b, self.c)

    def length(self) -> tuple:
        return distance(self.a, self.b), distance(self.b, self.c), distance(self.c, self.a)

    def perimeter(self) -> float:
        return sum(self.length())

    def area(self) -> float:
        return abs((self.b[0] - self.a[0]) * (self.c[1] - self.a[1])
                   - (self.b[1] - self.a[1]) * (self.c[0] - self.a[0])) / 2

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
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and isinstance(
                c, (list, tuple)) and isinstance(d, (list, tuple)):
            if len(a) == len(b) == len(c) == len(d) == 2:
                self.a, self.b, self.c, self.d = a, b, c, d
            else:
                raise ValueError("The coordinate value length can only be two")
        else:
            raise TypeError("The type of coordinate value can only be an array")

    def __repr__(self) -> str:
        return "Quadrilateral({}, {}, {}, {})".format(self.a, self.b, self.c, self.d)

    def length(self) -> tuple:
        return distance(self.a, self.b), distance(self.b, self.c), distance(self.c, self.d), distance(self.d, self.a)

    def perimeter(self) -> float:
        return sum(self.length())

    def area(self) -> float:
        return abs((self.b[0] - self.a[0]) * (self.c[1] - self.a[1]) - (self.b[1] - self.a[1]) * (
                self.c[0] - self.a[0]) - (self.d[0] - self.a[0]) * (self.c[1] - self.a[1]) + (
                           self.d[1] - self.a[1]) * (self.c[0] - self.a[0])) / 2

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

    def __slopes(self):
        return [Line(self.a, self.b).expr()[0], Line(self.c, self.d).expr()[0], Line(self.b, self.c).expr()[0],
                Line(self.d, self.a).expr()[0]]

    def is_trapezoidal(self, error: real = 0) -> bool:
        slopes = self.__slopes()
        slopes = [_ if _ is not None else 0 for _ in slopes]
        return True if abs(slopes[0] - slopes[1]) <= error or abs(slopes[2] - slopes[3]) <= error else False

    def is_parallelogram(self, error: real = 0) -> bool:
        slopes = self.__slopes()
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
            if isinstance(item, (list, tuple)):
                if len(item) != 2:
                    raise ValueError("The coordinate value length can only be two")
            else:
                raise TypeError("The type of coordinate value can only be an array")
        self.points = p

    def __repr__(self) -> str:
        return "Polygon(" + ", ".join(str(_) for _ in self.points) + ")"

    def length(self) -> tuple:
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
        if isinstance(center, (list, tuple)):
            if len(center) == 2:
                self.center = center
            else:
                raise ValueError("The coordinate value length can only be two")
        else:
            raise TypeError("The type of coordinate value can only be an array")
        if isinstance(radius, (int, float)):
            self.radius = radius
        else:
            raise TypeError("The type of length value can only be real numbers")

    def __repr__(self) -> str:
        return "Circle({}, {})".format(self.center, self.radius)

    def perimeter(self) -> float:
        return 6.283185307179586 * self.radius

    def area(self) -> float:
        return 3.141592653589793 * self.radius * self.radius

    def expr(self) -> list:
        return [self.center[0], self.center[1], self.radius]

    def chord(self, radian: real) -> float:
        from math import sin
        return 2 * self.radius * sin(radian / 2)


geom = Point, Line, Triangle, Quadrilateral, Polygon, Circle


def distance(g1, g2, error: real = 0) -> float:
    if isinstance(g1, (list, tuple)) and isinstance(g2, (list, tuple)):
        return ((g1[0] - g2[0]) ** 2 + (g1[1] - g2[1]) ** 2) ** 0.5
    if isinstance(g1, (list, tuple)):
        g1 = Point(g1)
    if isinstance(g2, (list, tuple)):
        g2 = Point(g2)
    if isinstance(g1, geom) and isinstance(g2, geom):
        if geom.index(type(g2)) < geom.index(type(g1)):
            return distance(g2, g1)
        elif isinstance(g1, Point) and isinstance(g2, Point):
            p1, p2 = g1.p, g2.p
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        elif isinstance(g1, Point) and isinstance(g2, Line):
            return Line(g1.p, g2.vertical(g1.p)).length()
        elif isinstance(g1, Line) and isinstance(g2, Line):
            e1, e2 = g1.expr(), g2.expr()
            if e1 == e2 == [None, None]:
                return abs(g1.a[0] - g2.a[0])
            elif abs(e1[0] - e2[0]) <= error:
                return abs(e1[1] - e2[1]) / (((e1[0] + e2[0]) / 2) ** 2 + 1) ** 0.5
            return 0
        elif isinstance(g1, Point) and isinstance(g2, Triangle):
            to_vertexes = [distance(g1.p, g2.a), distance(g1.p, g2.b), distance(g1.p, g2.c)]
            points = [Line(g2.a, g2.b).vertical(g1.p), Line(g2.b, g2.c).vertical(g1.p), Line(g2.c, g2.a).vertical(g1.p)]
            to_sides = [distance(g1.p, _) for _ in points]
            minimum = min(min(to_vertexes), min(to_sides))
            if min(to_sides) < min(to_vertexes):
                def determine():
                    index = to_sides.index(min(to_sides))
                    if (index == 0 and min(g2.a[0], g2.b[0]) < points[index][0] < max(g2.a[0], g2.b[0])
                        and min(g2.a[1], g2.b[1]) < points[index][1] < max(g2.a[1], g2.b[1])) or (
                            index == 1 and min(g2.b[0], g2.c[0]) < points[index][0] < max(g2.b[0], g2.c[0])
                            and min(g2.b[1], g2.c[1]) < points[index][1] < max(g2.b[1], g2.c[1])) or (
                            index == 2 and min(g2.c[0], g2.a[0]) < points[index][0] < max(g2.c[0], g2.a[0])
                            and min(g2.c[1], g2.a[1]) < points[index][1] < max(g2.c[1], g2.a[1])):
                        return True
                    return False

                if determine():
                    return minimum
                else:
                    to_sides[to_sides.index(min(to_sides))] = float("inf")
                    minimum = min(min(to_vertexes), min(to_sides))
                    if determine():
                        return minimum
                    to_sides[to_sides.index(min(to_sides))] = float("inf")
                    minimum = min(min(to_vertexes), min(to_sides))
            return minimum
        elif isinstance(g1, Line) and isinstance(g2, Triangle):
            pass
        elif isinstance(g1, Triangle) and isinstance(g2, Triangle):
            ...
        raise NotImplementedError
    raise TypeError("The input parameters can only be geometric shapes")
