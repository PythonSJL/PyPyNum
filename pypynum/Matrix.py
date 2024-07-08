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
        return Matrix([[self.data[i][j] * other.data[m][n] for j in range(self.cols) for n in range(other.cols)]
                       for i in range(self.rows) for m in range(other.rows)])

    def inner(self, other):
        if self.cols != other.cols:
            raise MatchError
        return Matrix([[sum([self.data[i][k] * other.data[j][k] for k in range(self.cols)]) for j in range(other.rows)]
                       for i in range(self.rows)])

    def outer(self, other):
        vec1, vec2 = self.flatten(), other.flatten()
        return Matrix([[a * b for b in vec2] for a in vec1])

    def t(self):
        return Matrix(list(map(list, zip(*self.data))))

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

    def slogdet(self):
        from math import log
        det_value = self.det()
        return (0.0, float("-inf")) if det_value == 0 else (det_value / abs(det_value), log(abs(det_value)))

    def inv(self):
        if self.rows != self.cols:
            raise SquareError
        n = self.rows
        augmented = [self[i, :] + [0 if k != i else 1 for k in range(n)] for i in range(n)]
        for i in range(n):
            max_el = abs(augmented[i][i])
            max_row = i
            for k in range(i + 1, n):
                if abs(augmented[k][i]) > max_el:
                    max_el = abs(augmented[k][i])
                    max_row = k
            if max_el == 0:
                raise ValueError("Matrix is singular and cannot be inverted")
            if max_row != i:
                augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
            pivot = augmented[i][i]
            for j in range(n * 2):
                augmented[i][j] /= pivot
            for j in range(n):
                if j != i:
                    factor = augmented[j][i]
                    for k in range(n * 2):
                        augmented[j][k] -= factor * augmented[i][k]
        return Matrix([augmented[i][n:] for i in range(n)])

    def rref(self, pivots=True):
        pe = []
        matrix = self.copy().data
        lead = 0
        for r in range(self.rows):
            if lead >= self.cols:
                return (Matrix(matrix), pe) if pivots else Matrix(matrix)
            i = r
            while matrix[i][lead] == 0:
                i += 1
                if i == self.rows:
                    i = r
                    lead += 1
                    if self.cols == lead:
                        return (Matrix(matrix), pe) if pivots else Matrix(matrix)
            matrix[i], matrix[r] = matrix[r], matrix[i]
            pe.append(lead)
            lv = matrix[r][lead]
            matrix[r] = [mrx / lv for mrx in matrix[r]]
            for i in range(self.rows):
                if i != r:
                    lv = matrix[i][lead]
                    matrix[i] = [iv - lv * rv for rv, iv in zip(matrix[r], matrix[i])]
            lead += 1
        return (Matrix(matrix), pe) if pivots else Matrix(matrix)

    def rank(self):
        matrix = self.rref(False).data
        for _ in range(self.rows):
            if matrix[_].count(0) == self.cols:
                return _
        return self.rows

    def diag(self):
        if self.cols > 1:
            return Matrix([[self.data[i][i]] for i in range(self.rows)])
        else:
            result = [[0] * self.rows for _ in range(self.rows)]
            for i in range(self.rows):
                result[i][i] = self.data[i][0]
            return Matrix(result)

    def i(self):
        if self.rows != self.cols:
            raise SquareError
        result = [[0] * self.cols for _ in range(self.rows)]
        for i in range(self.cols):
            result[i][i] = 1
        return Matrix(result)

    def norm(self, p=None):
        if p is None:
            return sum([sum([item * item for item in row]) for row in self.data]) ** 0.5
        elif p == 1:
            return max([sum([abs(item) for item in row]) for row in self.t().data])
        elif p == 2:
            return svd(self)[1][0, 0]
        elif p == float("inf"):
            return max([sum([abs(item) for item in row]) for row in self.data])
        else:
            raise ValueError("Invalid norm order of matrix")

    def normalize(self):
        norm = self.norm()
        return Matrix([[item / norm for item in row] for row in self.data])

    determinant = det
    diagonal = diag
    identity = i
    inverse = inv
    transpose = t
    trace = tr

    def __setitem__(self, key, value):
        if isinstance(key, (int, slice)):
            self.data[key] = value
        elif isinstance(key, tuple):
            if len(key) == 2:
                if isinstance(key[0], int) and isinstance(key[1], (int, slice)):
                    self.data[key[0]][key[1]] = value
                elif isinstance(key[0], slice) and isinstance(key[1], (int, slice)):
                    for old, new in zip(self.data[key[0]], value[key[0]]):
                        old[key[1]] = new[key[1]]
                else:
                    raise TypeError("The type of index is invalid")
            else:
                raise ValueError("The length of the index can only be one or two")
        else:
            raise TypeError("The type of index is invalid")


def mat(data):
    return Matrix(data)


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


def rank_decomp(matrix: Matrix) -> tuple:
    f, pivot_cols = matrix.rref()
    rank = len(pivot_cols)
    c = matrix[range(matrix.rows), pivot_cols]
    f = f[:rank, :]
    return Matrix(c, False), Matrix(f, False)


def cholesky(matrix, hermitian=True):
    rows = matrix.rows
    cols = matrix.cols
    if rows != cols:
        raise SquareError
    lower = [[0.0] * cols for _ in range(rows)]
    if hermitian:
        for i in range(matrix.rows):
            for j in range(i):
                lower[i][j] = (matrix[i][j] - sum([lower[i][k] * lower[j][k].conjugate()
                                                   for k in range(j)])) / lower[j][j]
            lower[i][i] = (matrix[i][i] - sum([lower[i][k] * lower[i][k].conjugate() for k in range(i)])) ** 0.5
    else:
        for i in range(matrix.rows):
            for j in range(i):
                lower[i][j] = (matrix[i][j] - sum([lower[i][k] * lower[j][k] for k in range(j)])) / lower[j][j]
            lower[i][i] = (matrix[i][i] - sum([lower[i][k] ** 2 for k in range(i)])) ** 0.5
    return Matrix(lower, False)


def lu(matrix: Matrix) -> tuple:
    rows = matrix.rows
    cols = matrix.cols
    if rows != cols:
        raise SquareError
    L = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        L[i][i] = 1.0
    U = matrix[:, :]
    for i in range(rows - 1):
        for j in range(i + 1, rows):
            L[j][i] = U[j][i] / U[i][i]
        for j in range(i + 1, rows):
            for k in range(i + 1, rows):
                U[j][k] -= L[j][i] * U[i][k]
        for j in range(i + 1, rows):
            U[j][i] = 0.0
    return Matrix(L), Matrix(U)


def tril_indices(n: int, k: int = 0, m: int = None) -> tuple:
    if m is None:
        m = n
    if n < 1 or k <= -n or m < 1:
        return [], []
    indices = [(y, x) for y in range(n) for x in range(y + k + 1) if x < m]
    return tuple(zip(*indices))


def identity(n: int) -> Matrix:
    data = [[0.0] * n for _ in range(n)]
    for i in range(n):
        data[i][i] = 1.0
    return Matrix(data, False)


def qr(matrix: Matrix) -> tuple:
    from math import hypot
    r = matrix.rows
    c = matrix.cols
    Q = identity(r)
    R = matrix
    rows, cols = tril_indices(r, -1, c)
    for row, col in zip(rows, cols):
        if R[row][col] != 0:
            _r = hypot(R[col][col], R[row][col])
            c = R[col][col] / _r
            s = -R[row][col] / _r
            G = identity(r)
            G[col][col] = c
            G[row][row] = c
            G[row][col] = s
            G[col][row] = -s
            Q = Q.inner(G)
            R = G @ R
    return Q, R


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
