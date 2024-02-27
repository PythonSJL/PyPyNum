from .Array import Array
from .errors import ShapeError

MatchError = ShapeError("The shapes of two matrices do not match")
SquareError = ShapeError("Matrix must be square")


class Matrix(Array):
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


def lu(matrix):
    rows = matrix.rows
    cols = matrix.cols
    if rows != cols:
        raise SquareError
    L = [[1.0 if a == b else 0.0 for a in range(cols)] for b in range(rows)]
    U = [[float(matrix.data[b][a]) for a in range(cols)] for b in range(rows)]
    for i in range(rows - 1):
        for j in range(i + 1, rows):
            L[j][i] = U[j][i] / U[i][i]
        for j in range(i + 1, rows):
            for k in range(i + 1, rows):
                U[j][k] = U[j][k] - L[j][i] * U[i][k]
        for j in range(i + 1, rows):
            U[j][i] = 0
    return Matrix(L), Matrix(U)


def tril_indices(n, k=0, m=None):
    if m is None:
        m = n
    if n < 1 or k <= -n or m < 1:
        return [], []
    _indices = [[x, y] for y in range(n) for x in range(y + k + 1) if x < m]
    return [y[1] for y in _indices], [x[0] for x in _indices]


def identity(n):
    return Matrix([[1.0 if a == b else 0.0 for a in range(n)] for b in range(n)])


def qr(matrix):
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


def eig(matrix):
    try:
        from numpy.linalg import eig as eigen
        e, Q = eigen(matrix.data)
        e, Q = Matrix([e.tolist()]), Matrix(Q.tolist())
        return e, Q
    except ImportError:
        eigen = "\n\033[91m提示：Matrix模块的eig函数可能存在计算错误\n\nTip: The eig function of the Matrix module may have calculation errors\033[m\n"
        print(eigen)
    _qr = []
    n = matrix.rows
    Q = identity(n)
    for i in range(100):
        _qr = qr(matrix)
        Q = Q @ _qr[0]
        matrix = _qr[1] @ _qr[0]
    AK = (_qr[0] @ _qr[1]).data
    e = Matrix([[AK[i][i] for i in range(n)]])
    return e, Q


def svd(matrix):
    e0, U = eig(matrix @ matrix.t())
    e1, V = eig(matrix.t() @ matrix)
    sigma = Matrix([[((e0.data[0][a] + e1.data[0][a]) / 2) ** 0.5 if a == b else 0 for a in range(matrix.cols)]
                    for b in range(matrix.rows)])
    return U, sigma, V.t()


del Array
