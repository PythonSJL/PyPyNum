from .types import arr, real


def _rotate_point(p, theta, cx, cy):
    from math import cos, sin
    x, y = p
    return [cx + (x - cx) * cos(theta) - (y - cy) * sin(theta), cy + (x - cx) * sin(theta) + (y - cy) * cos(theta)]


def _scale_point(p, k, cx, cy):
    x, y = p
    return [cx + (x - cx) * k, cy + (y - cy) * k]


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

    def translate(self, dx: real, dy: real):
        self.p = [self.p[0] + dx, self.p[1] + dy]
        return self

    def rotate(self, theta: real, center: arr = (0, 0)):
        self.p = _rotate_point(self.p, theta, center[0], center[1])
        return self

    def scale(self, k: real, center: arr = (0, 0)):
        self.p = _scale_point(self.p, k, center[0], center[1])
        return self


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

    def translate(self, dx: real, dy: real):
        self.a = [self.a[0] + dx, self.a[1] + dy]
        self.b = [self.b[0] + dx, self.b[1] + dy]
        return self

    def rotate(self, theta: real, center: arr = (0, 0)):
        self.a = _rotate_point(self.a, theta, center[0], center[1])
        self.b = _rotate_point(self.b, theta, center[0], center[1])
        return self

    def scale(self, k: real, center: arr = (0, 0)):
        self.a = _scale_point(self.a, k, center[0], center[1])
        self.b = _scale_point(self.b, k, center[0], center[1])
        return self

    def length(self) -> float:
        return ((self.a[0] - self.b[0]) ** 2 + (self.a[1] - self.b[1]) ** 2) ** 0.5

    def expr(self) -> list:
        if self.b[0] - self.a[0] == 0:
            k = None
        else:
            k = (self.b[1] - self.a[1]) / (self.b[0] - self.a[0])
        if k is None:
            b = self.a[0]
        else:
            b = self.a[1] - k * self.a[0]
        return [k, b]

    def project(self, p: arr) -> tuple:
        if isinstance(p, (list, tuple)):
            if len(p) == 2:
                k, b = self.expr()
                if k == 0:
                    x = p[0]
                    y = b
                    return x, y
                if k is None:
                    x = b
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

    def translate(self, dx: real, dy: real):
        self.a = [self.a[0] + dx, self.a[1] + dy]
        self.b = [self.b[0] + dx, self.b[1] + dy]
        self.c = [self.c[0] + dx, self.c[1] + dy]
        return self

    def rotate(self, theta: real, center: arr = (0, 0)):
        self.a = _rotate_point(self.a, theta, center[0], center[1])
        self.b = _rotate_point(self.b, theta, center[0], center[1])
        self.c = _rotate_point(self.c, theta, center[0], center[1])
        return self

    def scale(self, k: real, center: arr = (0, 0)):
        self.a = _scale_point(self.a, k, center[0], center[1])
        self.b = _scale_point(self.b, k, center[0], center[1])
        self.c = _scale_point(self.c, k, center[0], center[1])
        return self

    def length(self) -> tuple:
        return distance(self.a, self.b), distance(self.b, self.c), distance(self.c, self.a)

    def perimeter(self) -> float:
        return sum(self.length())

    def area(self) -> float:
        return abs(
            (self.b[0] - self.a[0]) * (self.c[1] - self.a[1]) - (self.b[1] - self.a[1]) * (self.c[0] - self.a[0])) / 2

    def incenter(self) -> tuple:
        a, b, c = self.length()
        x = (b * self.a[0] + c * self.b[0] + a * self.c[0]) / (a + b + c)
        y = (b * self.a[1] + c * self.b[1] + a * self.c[1]) / (a + b + c)
        return x, y

    def centroid(self) -> tuple:
        x = (self.a[0] + self.b[0] + self.c[0]) / 3
        y = (self.a[1] + self.b[1] + self.c[1]) / 3
        return x, y

    def circumcenter(self) -> tuple:
        x1, y1 = self.a
        x2, y2 = self.b
        x3, y3 = self.c
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if d == 0:
            raise ValueError("Degenerate triangle has no circumcenter")
        ux = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (y1 - y2)) / d
        uy = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (x2 - x1)) / d
        return ux, uy

    def orthocenter(self) -> tuple:
        if self.area() == 0:
            raise ValueError("Degenerate triangle has no orthocenter")
        ox, oy = self.circumcenter()
        return self.a[0] + self.b[0] + self.c[0] - 2 * ox, self.a[1] + self.b[1] + self.c[1] - 2 * oy

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

    def translate(self, dx: real, dy: real):
        self.a = [self.a[0] + dx, self.a[1] + dy]
        self.b = [self.b[0] + dx, self.b[1] + dy]
        self.c = [self.c[0] + dx, self.c[1] + dy]
        self.d = [self.d[0] + dx, self.d[1] + dy]
        return self

    def rotate(self, theta: real, center: arr = (0, 0)):
        self.a = _rotate_point(self.a, theta, center[0], center[1])
        self.b = _rotate_point(self.b, theta, center[0], center[1])
        self.c = _rotate_point(self.c, theta, center[0], center[1])
        self.d = _rotate_point(self.d, theta, center[0], center[1])
        return self

    def scale(self, k: real, center: arr = (0, 0)):
        self.a = _scale_point(self.a, k, center[0], center[1])
        self.b = _scale_point(self.b, k, center[0], center[1])
        self.c = _scale_point(self.c, k, center[0], center[1])
        self.d = _scale_point(self.d, k, center[0], center[1])
        return self

    def length(self) -> tuple:
        return distance(self.a, self.b), distance(self.b, self.c), distance(self.c, self.d), distance(self.d, self.a)

    def perimeter(self) -> float:
        return sum(self.length())

    def area(self) -> float:
        return abs(
            (self.b[0] - self.a[0]) * (self.c[1] - self.a[1]) - (self.b[1] - self.a[1]) * (self.c[0] - self.a[0]) - (
                    self.d[0] - self.a[0]) * (self.c[1] - self.a[1]) + (self.d[1] - self.a[1]) * (
                    self.c[0] - self.a[0])) / 2

    def centroid(self) -> tuple:
        return Polygon(self.a, self.b, self.c, self.d).centroid()

    @staticmethod
    def __is_parallel(k1, k2, error: real = 0) -> bool:
        if k1 is None and k2 is None:
            return True
        if k1 is None or k2 is None:
            return False
        return abs(k1 - k2) <= error

    def __slopes(self):
        return [Line(self.a, self.b).expr()[0], Line(self.c, self.d).expr()[0], Line(self.b, self.c).expr()[0],
                Line(self.d, self.a).expr()[0]]

    def is_trapezoidal(self, error: real = 0) -> bool:
        slopes = self.__slopes()
        return self.__is_parallel(slopes[0], slopes[1], error) or self.__is_parallel(slopes[2], slopes[3], error)

    def is_parallelogram(self, error: real = 0) -> bool:
        slopes = self.__slopes()
        return self.__is_parallel(slopes[0], slopes[1], error) and self.__is_parallel(slopes[2], slopes[3], error)

    def is_diamond(self, error: real = 0) -> bool:
        _length = self.length()
        return True if abs(_length[0] - _length[1]) <= error and abs(_length[1] - _length[2]) <= error and abs(
            _length[2] - _length[3]) <= error and abs(_length[3] - _length[0]) <= error else False

    def is_rectangular(self, error: real = 0) -> bool:
        if not self.is_parallelogram(error):
            return False
        slopes = self.__slopes()
        k1, k2 = slopes[0], slopes[2]
        if k1 is None and k2 is not None and abs(k2) <= error:
            return True
        if k2 is None and k1 is not None and abs(k1) <= error:
            return True
        if k1 is None or k2 is None:
            return False
        return abs(k1 * k2 + 1) <= error

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
        return "Polygon(" + ", ".join(map(str, self.points)) + ")"

    def translate(self, dx: real, dy: real):
        self.points = tuple([[p[0] + dx, p[1] + dy] for p in self.points])
        return self

    def rotate(self, theta: real, center: arr = (0, 0)):
        self.points = tuple([_rotate_point(p, theta, center[0], center[1]) for p in self.points])
        return self

    def scale(self, k: real, center: arr = (0, 0)):
        self.points = tuple([_scale_point(p, k, center[0], center[1]) for p in self.points])
        return self

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
        _area = 0
        cx = 0
        cy = 0
        q = self.points[-1]
        for p in self.points:
            cross = p[0] * q[1] - p[1] * q[0]
            _area += cross
            cx += (p[0] + q[0]) * cross
            cy += (p[1] + q[1]) * cross
            q = p
        _area /= 2
        if _area == 0:
            raise ValueError("Degenerate polygon has no centroid")
        cx /= 6 * _area
        cy /= 6 * _area
        return cx, cy


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

    def translate(self, dx: real, dy: real):
        self.center = [self.center[0] + dx, self.center[1] + dy]
        return self

    def rotate(self, theta: real, center: arr = (0, 0)):
        self.center = _rotate_point(self.center, theta, center[0], center[1])
        return self

    def scale(self, k: real, center: arr = (0, 0)):
        self.center = _scale_point(self.center, k, center[0], center[1])
        self.radius *= abs(k)
        return self

    def perimeter(self) -> float:
        return 6.283185307179586 * self.radius

    def area(self) -> float:
        return 3.141592653589793 * self.radius * self.radius

    def expr(self) -> list:
        return [self.center[0], self.center[1], self.radius]

    def chord(self, radian: real) -> float:
        from math import sin
        return 2 * self.radius * sin(radian / 2)


def distance(g1, g2, error: real = 0) -> float:
    def is_point_inside_polygon(p, vertices, err: real = 0):
        n = len(vertices)
        total_area = 0
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]
            total_area += x1 * y2 - x2 * y1
        total_area = abs(total_area) / 2.0
        if total_area <= err:
            return False
        px, py = p
        split_area = 0
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]
            cross = (x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)
            split_area += abs(cross)
        split_area /= 2.0
        return split_area <= total_area + err

    def point_poly_dist(point_p, vertices):
        if is_point_inside_polygon(point_p, vertices, error):
            return 0.0
        to_vertexes = [distance(point_p, v) for v in vertices]
        edges = [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]
        points = [Line(e[0], e[1]).project(point_p) for e in edges]
        to_sides = [distance(point_p, _) for _ in points]

        def on_segment(idx):
            (x1, y1), (x2, y2) = edges[idx]
            px, py = points[idx]
            return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)

        min_vert = min(to_vertexes)
        for _ in range(len(vertices)):
            if min(to_sides) >= min_vert:
                break
            if on_segment(to_sides.index(min(to_sides))):
                return min(to_sides)
            to_sides[to_sides.index(min(to_sides))] = float("inf")
        return min_vert

    def line_poly_dist(line_l, vertices):
        k, b = line_l.expr()

        def get_side_val(p):
            if k is None:
                return p[0] - b
            return k * p[0] - p[1] + b

        n = len(vertices)
        min_vert_dist = float("inf")
        for i in range(n):
            _p1 = vertices[i]
            _p2 = vertices[(i + 1) % n]
            d1 = distance(line_l, _p1)
            d2 = distance(line_l, _p2)
            if d1 < min_vert_dist:
                min_vert_dist = d1
            if d2 < min_vert_dist:
                min_vert_dist = d2
            if d1 <= error or d2 <= error:
                return 0.0
            val1 = get_side_val(_p1)
            val2 = get_side_val(_p2)
            if val1 * val2 < 0:
                return 0.0
        return min_vert_dist

    def poly_poly_dist(verts1, verts2):
        min_dist = float("inf")
        for v in verts1:
            d = point_poly_dist(v, verts2)
            if d < min_dist:
                min_dist = d
        for v in verts2:
            d = point_poly_dist(v, verts1)
            if d < min_dist:
                min_dist = d
        n1 = len(verts1)
        for i in range(n1):
            line = Line(verts1[i], verts1[(i + 1) % n1])
            d = line_poly_dist(line, verts2)
            if d < min_dist:
                min_dist = d
        n2 = len(verts2)
        for i in range(n2):
            line = Line(verts2[i], verts2[(i + 1) % n2])
            d = line_poly_dist(line, verts1)
            if d < min_dist:
                min_dist = d
        return min_dist

    geom = Point, Line, Triangle, Quadrilateral, Polygon, Circle
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
            return Line(g1.p, g2.project(g1.p)).length()
        elif isinstance(g1, Line) and isinstance(g2, Line):
            e1, e2 = g1.expr(), g2.expr()
            k1, b1 = e1
            k2, b2 = e2
            if k1 is None and k2 is None:
                return abs(g1.a[0] - g2.a[0])
            if k1 is None or k2 is None:
                return 0
            if abs(k1 - k2) <= error:
                return abs(b1 - b2) / ((k1 ** 2 + 1) ** 0.5)
            return 0
        elif isinstance(g1, Point) and isinstance(g2, Triangle):
            return point_poly_dist(g1.p, (g2.a, g2.b, g2.c))
        elif isinstance(g1, Point) and isinstance(g2, Quadrilateral):
            return point_poly_dist(g1.p, (g2.a, g2.b, g2.c, g2.d))
        elif isinstance(g1, Point) and isinstance(g2, Polygon):
            return point_poly_dist(g1.p, g2.points)
        elif isinstance(g1, Line) and isinstance(g2, Triangle):
            return line_poly_dist(g1, (g2.a, g2.b, g2.c))
        elif isinstance(g1, Line) and isinstance(g2, Quadrilateral):
            return line_poly_dist(g1, (g2.a, g2.b, g2.c, g2.d))
        elif isinstance(g1, Line) and isinstance(g2, Polygon):
            return line_poly_dist(g1, g2.points)
        elif isinstance(g1, Triangle) and isinstance(g2, Triangle):
            return poly_poly_dist((g1.a, g1.b, g1.c), (g2.a, g2.b, g2.c))
        elif isinstance(g1, Triangle) and isinstance(g2, Quadrilateral):
            return poly_poly_dist((g1.a, g1.b, g1.c), (g2.a, g2.b, g2.c, g2.d))
        elif isinstance(g1, Triangle) and isinstance(g2, Polygon):
            return poly_poly_dist((g1.a, g1.b, g1.c), g2.points)
        elif isinstance(g1, Quadrilateral) and isinstance(g2, Quadrilateral):
            return poly_poly_dist((g1.a, g1.b, g1.c, g1.d), (g2.a, g2.b, g2.c, g2.d))
        elif isinstance(g1, Quadrilateral) and isinstance(g2, Polygon):
            return poly_poly_dist((g1.a, g1.b, g1.c, g1.d), g2.points)
        elif isinstance(g1, Polygon) and isinstance(g2, Polygon):
            return poly_poly_dist(g1.points, g2.points)
        elif isinstance(g1, Point) and isinstance(g2, Circle):
            d = distance(g1, g2.center)
            return max(0.0, d - g2.radius)
        elif isinstance(g1, Line) and isinstance(g2, Circle):
            d = distance(g1, g2.center)
            return max(0.0, d - g2.radius)
        elif isinstance(g1, Triangle) and isinstance(g2, Circle):
            d = point_poly_dist(g2.center, (g1.a, g1.b, g1.c))
            return max(0.0, d - g2.radius)
        elif isinstance(g1, Quadrilateral) and isinstance(g2, Circle):
            d = point_poly_dist(g2.center, (g1.a, g1.b, g1.c, g1.d))
            return max(0.0, d - g2.radius)
        elif isinstance(g1, Polygon) and isinstance(g2, Circle):
            d = point_poly_dist(g2.center, g1.points)
            return max(0.0, d - g2.radius)
        elif isinstance(g1, Circle) and isinstance(g2, Circle):
            d = distance(g1.center, g2.center)
            r_sum = g1.radius + g2.radius
            return max(0.0, d - r_sum)
    raise TypeError("The input parameters can only be geometric shapes")
