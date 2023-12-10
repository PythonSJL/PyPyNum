arr = list | tuple


class Line:
    def __init__(self, a: arr, b: arr):
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) == len(b) == 2:
                self.a, self.b = a, b
            else:
                raise ValueError("The coordinate value length can only be two")
        else:
            raise ValueError("The type of coordinate value can only be an array")

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
            else:
                raise ValueError("The coordinate value length can only be two")
        else:
            raise ValueError("The type of coordinate value can only be an array")


class Triangle:
    def __init__(self, a: arr, b: arr, c: arr):
        if isinstance(a, arr) and isinstance(b, arr) and isinstance(c, arr):
            if len(a) == len(b) == len(c) == 2:
                self.a, self.b, self.c = a, b, c
            else:
                raise ValueError("The coordinate value length can only be two")
        else:
            raise ValueError("The type of coordinate value can only be an array")

    def length(self) -> tuple:
        def distance(p1: arr, p2: arr) -> float:
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        return distance(self.a, self.b), distance(self.b, self.c), distance(self.c, self.a)

    def area(self) -> float:
        a, b, c = self.length()
        p = (a + b + c) / 2
        return (p * (p - a) * (p - b) * (p - c)) ** 0.5

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
        # FIXME ( There are some errors here )
        divisor = (a[0] * b[1] - a[0] * c[1] - a[1] * b[0] + a[1] * c[0] + b[0] * c[1] - b[1] * c[0])
        x = (-a[0] * a[1] * b[0] + a[0] * a[1] * c[0] + a[0] * b[0] * b[1] - a[0] * c[0] * c[1] + a[1] ** 2 * b[1] - a[
            1] ** 2 * c[1] - a[1] * b[1] ** 2 + a[1] * c[1] ** 2 - b[0] * b[1] * c[0] + b[0] * c[0] * c[1] + b[1] ** 2 *
             c[1] - b[1] * c[1] ** 2) / divisor
        y = (a[0] ** 2 * b[0] - a[0] ** 2 * c[0] - a[0] * a[1] * b[1] + a[0] * a[1] * c[1] - a[0] * b[0] ** 2 + a[0] *
             c[0] ** 2 + a[1] * b[0] * b[1] - a[1] * c[0] * c[1] + b[0] ** 2 * c[0] - b[0] * b[1] * c[1] - b[0] * c[
                 0] ** 2 + b[1] * c[0] * c[1]) / divisor
        return x, y
