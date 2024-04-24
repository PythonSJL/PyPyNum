from .Array import Array
from .errors import ShapeError

MatchError = ShapeError("The shapes of two matrices do not match")
SquareError = ShapeError("Matrix must be square")


class Matrix(Array):
    """
    It is a matrix class that supports basic operations,
    as well as functions such as determinant and inverse matrix.
    :param data: An array in the form of a list
    :param check: Check the rationality of the input array
    """

    def __init__(self, data, check=True):
        super().__init__(data, check)
        if len(self.shape) != 2:
            raise ShapeError("Matrix can only be two-dimensional in shape")
        self.rows = len(data)
        self.cols = len(data[0])

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise MatchError
        return Matrix([[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])

    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise MatchError
        return Matrix([[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise MatchError
            return Matrix([[self.data[i][j] * other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])
        else:
            return Matrix([[item * other for item in row] for row in self.data])

    def __matmul__(self, other):
        if self.cols != other.rows:
            raise MatchError
        return Matrix([[sum([self.data[i][k] * other.data[k][j] for k in range(self.cols)]) for j in range(other.cols)]
                       for i in range(self.rows)])

    def kron(self, other):
        result = [[0 for _ in range(self.cols * other.cols)] for _ in range(self.rows * other.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                for m in range(other.rows):
                    for n in range(other.cols):
                        result[i * other.rows + m][j * other.cols + n] = self.data[i][j] * other.data[m][n]
        return Matrix(result)

    def inner(self, other):
        return self @ other.t()

    def outer(self, other):
        return Matrix([[a] for b in self.data for a in b]) @ Matrix([[a for b in other.data for a in b]])

    def t(self):
        return Matrix([[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)])

    def minor(self, row, col):
        return Matrix([row[:col] + row[col + 1:] for row in self.data[:row] + self.data[row + 1:]])

    def tr(self):
        if self.rows != self.cols:
            raise SquareError
        return sum([self.data[i][i] for i in range(self.rows)])

    def det(self):
        if self.rows != self.cols:
            raise SquareError
        matrix = self.copy().data
        size = len(matrix)
        determinant = 1
        for i in range(size):
            if matrix[i][i] == 0:
                for j in range(i + 1, size):
                    if matrix[i][j] != 0:
                        matrix[i], matrix[j] = matrix[j], matrix[i]
                        determinant = -determinant
                        break
                if matrix[i][i] == 0:
                    return 0
            determinant *= matrix[i][i]
            for j in range(i + 1, size):
                matrix[i][j] /= matrix[i][i]
                for k in range(i + 1, size):
                    matrix[k][j] -= matrix[i][j] * matrix[k][i]
        return determinant

    def inv(self):
        if self.rows != self.cols:
            raise SquareError
        det = self.det()
        if det == 0:
            return None
        matrix_minor = [[Matrix(Matrix(self.data).minor(i, j).data).det() for j in range(self.cols)] for i in
                        range(self.rows)]
        cofactors = [[x * (-1) ** (row + col) for col, x in enumerate(matrix_minor[row])] for row in range(self.cols)]
        adjugate = Matrix(cofactors).t().data
        return Matrix(adjugate) * (1 / det)

    def rref(self):
        matrix = self.copy().data
        lead = 0
        for r in range(self.rows):
            if lead >= self.cols:
                return Matrix(matrix)
            i = r
            while matrix[i][lead] == 0:
                i += 1
                if i == self.rows:
                    i = r
                    lead += 1
                    if self.cols == lead:
                        return Matrix(matrix)
            matrix[i], matrix[r] = matrix[r], matrix[i]
            lv = matrix[r][lead]
            matrix[r] = [mrx / lv for mrx in matrix[r]]
            for i in range(self.rows):
                if i != r:
                    lv = matrix[i][lead]
                    matrix[i] = [iv - lv * rv for rv, iv in zip(matrix[r], matrix[i])]
            lead += 1
        return Matrix(matrix)

    def rank(self):
        matrix = self.rref().data
        for _ in range(self.rows):
            if matrix[_].count(0) == self.cols:
                return _
        return self.rows

    def diag(self):
        if self.cols > 1:
            return Matrix([[self.data[i][i]] for i in range(self.rows)])
        else:
            result = [[0 for _ in range(self.rows)] for _ in range(self.rows)]
            for i in range(self.rows):
                result[i][i] = self.data[i][0]
            return Matrix(result)

    def i(self):
        if self.rows != self.cols:
            raise SquareError
        result = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.cols):
            result[i][i] = 1
        return Matrix(result)

    def norm(self, p=None):
        if p is None:
            return sum([sum([item * item for item in row]) for row in self.data]) ** 0.5
        elif p == 1:
            return max([sum([abs(item) for item in row]) for row in self.t().data])
        elif p == 2:
            return svd(self)[1].data[0][0]
        elif p == float("inf"):
            return max([sum([abs(item) for item in row]) for row in self.data])
        else:
            raise ValueError("Invalid norm order of matrix")

    def normalize(self):
        norm = self.norm()
        return Matrix([[item / norm for item in row] for row in self.data])

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return self.data[item]
        elif isinstance(item, tuple):
            if len(item) == 2:
                if isinstance(item[0], int) and isinstance(item[1], (int, slice)):
                    return self.data[item[0]][item[1]]
                elif isinstance(item[0], slice) and isinstance(item[1], (int, slice)):
                    return [row[item[1]] for row in self.data[item[0]]]
                else:
                    return []
            else:
                raise ValueError("The length of the index can only be one or two")
        else:
            raise ValueError("The type of index is invalid")

    def __setitem__(self, key, value):
        if isinstance(key, (int, slice)):
            self.data[key] = value
        elif isinstance(key, tuple):
            if len(key) == 2:
                if isinstance(key[0], int) and isinstance(key[1], (int, slice)):
                    self.data[key[0]][key[1]] = value
                else:
                    raise NotImplementedError
            else:
                raise ValueError("The length of the index can only be one or two")
        else:
            raise ValueError("The type of index is invalid")


def zeros(_dimensions):
    from .Array import zeros as _zeros
    return Matrix(_zeros(_dimensions))


def zeros_like(_nested_list):
    from .Array import zeros_like as _zeros_like
    return Matrix(_zeros_like(_nested_list.data))


def mat(data):
    return Matrix(data)


def same(rows, cols, value=0):
    return Matrix([[value] * cols for _ in range(rows)])


def rotate90(matrix: Matrix, times: int) -> Matrix:
    matrix = matrix.copy()
    if times % 4 == 0:
        return matrix
    elif times % 4 == 1:
        return Matrix([list(col)[::-1] for col in zip(*matrix)])
    elif times % 4 == 2:
        return Matrix([row[::-1] for row in matrix[::-1]])
    else:
        return Matrix([list(col)[::-1] for col in zip(*matrix[::-1])][::-1])


def lu(matrix: Matrix) -> tuple:
    rows = matrix.rows
    cols = matrix.cols
    if rows != cols:
        raise SquareError
    L = [[1.0 if a == b else 0.0 for a in range(cols)] for b in range(rows)]
    U = [[float(matrix[b][a]) for a in range(cols)] for b in range(rows)]
    for i in range(rows - 1):
        for j in range(i + 1, rows):
            L[j][i] = U[j][i] / U[i][i]
        for j in range(i + 1, rows):
            for k in range(i + 1, rows):
                U[j][k] = U[j][k] - L[j][i] * U[i][k]
        for j in range(i + 1, rows):
            U[j][i] = 0
    return Matrix(L), Matrix(U)


def tril_indices(n: int, k: int = 0, m: int = None) -> tuple:
    if m is None:
        m = n
    if n < 1 or k <= -n or m < 1:
        return [], []
    _indices = [[x, y] for y in range(n) for x in range(y + k + 1) if x < m]
    return [y[1] for y in _indices], [x[0] for x in _indices]


def identity(n: int) -> Matrix:
    return Matrix([[1.0 if a == b else 0.0 for a in range(n)] for b in range(n)])


def qr(matrix: Matrix) -> tuple:
    from math import hypot
    r = matrix.rows
    c = matrix.cols
    Q = identity(r).data
    R = [[float(item) for item in line] for line in matrix.data]
    rows, cols = tril_indices(r, -1, c)
    for row, col in zip(rows, cols):
        if R[row][col] != 0:
            _r = hypot(R[col][col], R[row][col])
            c = R[col][col] / _r
            s = -R[row][col] / _r
            G = identity(r).data
            G[col][col] = c
            G[row][row] = c
            G[row][col] = s
            G[col][row] = -s
            G = Matrix(G)
            Q = Matrix(Q)
            R = Matrix(R)
            Q = Q.inner(G).data
            R = (G @ R).data
    return Matrix([[item for item in line] for line in Q]), Matrix([[item for item in line] for line in R])


def hessenberg(matrix: Matrix) -> tuple:
    from math import sqrt
    a = matrix.copy()
    n = a.rows
    t = [0.0] * n
    a[-1, -1] = float(a[-1, -1])
    if n > 2:
        inf = float("inf")
        for i in range(n - 1, 1, -1):
            scale = 0.0
            for k in range(0, i):
                scale += abs(a[i, k].real) + abs(a[i, k].imag)
            scale_inv = 0.0
            if scale != 0:
                scale_inv = 1 / scale
            if scale == 0 or scale_inv == inf:
                t[i] = 0.0
                a[i, i - 1] = 0.0
                continue
            h = 0.0
            for k in range(0, i):
                a[i, k] = a[i, k] * scale_inv
                rr, ii = a[i, k].real, a[i, k].imag
                h += rr * rr + ii * ii
            f = a[i, i - 1]
            f0 = abs(f)
            g = sqrt(h)
            a[i, i - 1] = - g * scale
            if f0 == 0:
                t[i] = g
            else:
                ff = f / f0
                t[i] = f + g * ff
                a[i, i - 1] = a[i, i - 1] * ff
            h += g * f0
            h = 1 / sqrt(h)
            t[i] *= h
            for k in range(0, i - 1):
                a[i, k] = a[i, k] * h
            for j in range(0, i):
                g = t[i].conjugate() * a[j, i - 1]
                for k in range(0, i - 1):
                    g += a[i, k].conjugate() * a[j, k]
                a[j, i - 1] = a[j, i - 1] - g * t[i]
                for k in range(0, i - 1):
                    a[j, k] = a[j, k] - g * a[i, k]
            for j in range(0, n):
                g = t[i] * a[i - 1, j]
                for k in range(0, i - 1):
                    g += a[i, k] * a[k, j]
                a[i - 1, j] = a[i - 1, j] - g * t[i].conjugate()
                for k in range(0, i - 1):
                    a[k, j] = a[k, j] - g * a[i, k].conjugate()
    q = a.copy()
    if n == 1:
        q[0, 0] = 1.0
        return q, a
    q[0, 0] = q[1, 1] = 1.0
    q[0, 1] = q[1, 0] = 0.0
    for i in range(2, n):
        if t[i] != 0:
            for j in range(0, i):
                g = t[i] * q[i - 1, j]
                for k in range(0, i - 1):
                    g += q[i, k] * q[k, j]
                q[i - 1, j] -= g * t[i].conjugate()
                for k in range(0, i - 1):
                    q[k, j] -= g * a[i, k].conjugate()
        q[i, i] = 1.0
        for j in range(0, i):
            q[j, i] = q[i, j] = 0.0
    for x in range(n):
        for y in range(x + 2, n):
            a[y, x] = 0.0
    return q, a


def eigen(matrix: Matrix) -> tuple:
    from math import hypot
    from cmath import sqrt
    a = matrix.copy()
    n = a.rows
    if n == 1:
        return Matrix([[float(a[0, 0])]]), Matrix([[1.0]])
    q, a = hessenberg(a)
    norm = 0
    for x in range(n):
        for y in range(min(x + 2, n)):
            norm += abs(a[y, x])
    norm = norm ** 0.5 / n
    if norm == 0:
        return None, None
    n0 = 0
    n1 = n
    its = 0
    while True:
        k = n0
        while k + 1 < n1:
            if abs(a[k + 1, k]) <= 1e-100:
                break
            k += 1
        if k + 1 < n1:
            a[k + 1, k] = 0.0
            n0 = k + 1
            its = 0
            if n0 + 1 >= n1:
                n0 = 0
                n1 = k + 1
                if n1 < 2:
                    break
        else:
            if its % 30 == 10:
                shift = a[n1 - 1, n1 - 2]
            elif its % 30 == 20:
                shift = abs(a[n1 - 1, n1 - 2])
            elif its % 30 == 29:
                shift = norm
            else:
                t = a[n1 - 2, n1 - 2] + a[n1 - 1, n1 - 1]
                s = (a[n1 - 1, n1 - 1] - a[n1 - 2, n1 - 2]) ** 2 + 4 * a[n1 - 1, n1 - 2] * a[n1 - 2, n1 - 1]
                if s.real > 0:
                    s = sqrt(s)
                else:
                    s = sqrt(-s) * 1j
                if s.imag == 0:
                    s = s.real
                a0 = (t + s) / 2
                b0 = (t - s) / 2
                if abs(a[n1 - 1, n1 - 1] - a0) > abs(a[n1 - 1, n1 - 1] - b0):
                    shift = b0
                else:
                    shift = a0
            its += 1
            c = a[n0, n0] - shift
            s = a[n0 + 1, n0]
            v = hypot(abs(c), abs(s))
            if v == 0:
                c = 1
                s = 0
            else:
                c /= v
                s /= v
            cc = c.conjugate()
            cs = s.conjugate()
            for k in range(n0, n):
                x = a[n0, k]
                y = a[n0 + 1, k]
                a[n0, k] = cc * x + cs * y
                a[n0 + 1, k] = c * y - s * x
            for k in range(min(n1, n0 + 3)):
                x = a[k, n0]
                y = a[k, n0 + 1]
                a[k, n0] = c * x + s * y
                a[k, n0 + 1] = cc * y - cs * x
            if not isinstance(q, bool):
                for k in range(n):
                    x = q[k, n0]
                    y = q[k, n0 + 1]
                    q[k, n0] = c * x + s * y
                    q[k, n0 + 1] = cc * y - cs * x
            for j in range(n0, n1 - 2):
                c = a[j + 1, j]
                s = a[j + 2, j]
                v = hypot(abs(c), abs(s))
                if v == 0:
                    a[j + 1, j] = 0.0
                    c = 1
                    s = 0
                else:
                    a[j + 1, j] = v
                    c /= v
                    s /= v
                a[j + 2, j] = 0.0
                cc = c.conjugate()
                cs = s.conjugate()
                for k in range(j + 1, n):
                    x = a[j + 1, k]
                    y = a[j + 2, k]
                    a[j + 1, k] = cc * x + cs * y
                    a[j + 2, k] = c * y - s * x
                for k in range(0, min(n1, j + 4)):
                    x = a[k, j + 1]
                    y = a[k, j + 2]
                    a[k, j + 1] = c * x + s * y
                    a[k, j + 2] = cc * y - cs * x
                if not isinstance(q, bool):
                    for k in range(0, n):
                        x = q[k, j + 1]
                        y = q[k, j + 2]
                        q[k, j + 1] = c * x + s * y
                        q[k, j + 2] = cc * y - cs * x
    e = Matrix([[a[i, i] if j == i else 0 for j in range(n)] for i in range(n)])
    er = identity(n)
    m = 1
    for i in range(1, n):
        s = a[i, i]
        for j in range(i - 1, -1, -1):
            r = 0
            for k in range(j + 1, i + 1):
                r += a[j, k] * er[k, i]
            t = a[j, j] - s
            r = -r / t
            er[j, i] = r
            m = max(m, abs(r))
        if m != 1:
            for k in range(0, i + 1):
                er[k, i] /= m
    q = q @ er
    return e, q


def svd(matrix: Matrix) -> tuple:
    print("\n警告：SVD函数可能存在计算错误\n\nWarning: The SVD function may have calculation errors\n")
    e0, u = eigen(matrix @ matrix.t())
    u = Matrix([_[::-1] for _ in u])
    e1, v = eigen(matrix.t() @ matrix)
    sigma = Matrix([[((e0[i, i] + e1[i, i]) / 2) ** 0.5 if j == i else 0 for j in range(matrix.cols - 1, -1, -1)]
                    for i in range(matrix.rows - 1, -1, -1)])
    return u, sigma, v.t()


del Array
