from .arrays import Array
from .types import Any, Callable, ShapeError, arr, config

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

    def diag(self, k=0):
        return diag(self, k)

    def i(self):
        return identity(self.rows, self.cols)

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

    def latex(self, matrix_type="bmatrix", row_spacing=0, col_spacing=0, spacing_unit="mm"):
        valid_matrix_types = ["Bmatrix", "Vmatrix", "array", "bmatrix", "matrix", "pmatrix", "smallmatrix", "vmatrix"]
        valid_units = ["bp", "cc", "cm", "dd", "em", "ex", "in", "mm", "mu", "pc", "pt", "sp"]
        if matrix_type not in valid_matrix_types:
            raise ValueError("Invalid matrix type. Choose from {}".format(valid_matrix_types))
        if spacing_unit not in valid_units:
            raise ValueError("Invalid spacing unit. Choose from {}".format(valid_units))
        alignment = "c" * len(self[0]) if matrix_type == "array" else None
        col_separator = " & " if col_spacing == 0 else " & \\hspace{{{}{}}} & ".format(col_spacing, spacing_unit)
        row_separator = "\\\\" if row_spacing == 0 else "\\\\[{}{}]".format(row_spacing, spacing_unit)
        latex_str = ["\\begin{{{}}}".format(matrix_type), "{{{}}}".format(alignment) if alignment else "",
                     row_separator.join([col_separator.join(map(str, row)) for row in self]),
                     "\\end{{{}}}".format(matrix_type)]
        return "".join(latex_str)

    def __str__(self, use_latex=False):
        use_latex = config.use_latex or use_latex
        return self.latex() if use_latex else super().__str__()

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
        return Matrix([list(col)[::-1] for col in zip(*matrix)], False)
    elif times % 4 == 2:
        return Matrix([row[::-1] for row in matrix[::-1]], False)
    else:
        return Matrix([list(col)[::-1] for col in zip(*matrix[::-1])][::-1], False)


def rank_decomp(matrix: Matrix) -> tuple:
    f, pivot_cols = matrix.rref()
    rank = len(pivot_cols)
    c = matrix[range(matrix.rows), pivot_cols]
    f = f[:rank, :]
    return Matrix(c, False), Matrix(f, False)


def cholesky(matrix: Matrix, hermitian: bool = True) -> Matrix:
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
    lower = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        lower[i][i] = 1.0
    upper = matrix[:, :]
    for i in range(rows - 1):
        for j in range(i + 1, rows):
            lower[j][i] = upper[j][i] / upper[i][i]
        for j in range(i + 1, rows):
            for k in range(i + 1, rows):
                upper[j][k] -= lower[j][i] * upper[i][k]
        for j in range(i + 1, rows):
            upper[j][i] = 0.0
    return Matrix(lower, False), Matrix(upper, False)


def triu_indices(n: int, k: int = 0, m: int = None) -> tuple:
    if m is None:
        m = n
    if n < 1 or k >= m or m < 1:
        return (), ()
    indices = [(y, x) for y in range(n) for x in range(y + k, m) if x > -1]
    return tuple(zip(*indices))


def tril_indices(n: int, k: int = 0, m: int = None) -> tuple:
    if m is None:
        m = n
    if n < 1 or k <= -n or m < 1:
        return (), ()
    indices = [(y, x) for y in range(n) for x in range(y + k + 1) if x < m]
    return tuple(zip(*indices))


def diag_indices(n: int, k: int = 0, m: int = None) -> tuple:
    if m is None:
        m = n
    if n < 1 or k <= -n or k >= m or m < 1:
        return (), ()
    indices = [(i, i + k) for i in range(min(m - k, n))] if k >= 0 else [(i - k, i) for i in range(min(n + k, m))]
    return tuple(zip(*indices))


def diag(v: Any, k: int = 0, n: int = None, m: int = None) -> Any:
    if isinstance(v, Matrix):
        rows, cols = diag_indices(v.rows, k, v.cols)
        return [v[i, j] for i, j in zip(rows, cols)]
    else:
        if n is None and m is None:
            n = m = len(v) + abs(k)
        else:
            m = n
        result = [[0] * n for _ in range(n)]
        rows, cols = diag_indices(n, k, m)
        for row, col, val in zip(rows, cols, v):
            result[row][col] = val
        return Matrix(result, False)


def identity(n: int, m: int = None) -> Matrix:
    if m is None:
        m = n
    result = [[0] * m for _ in range(n)]
    minimum = min(n, m)
    for i in range(minimum):
        result[i][i] = 1
    return Matrix(result, False)


def perm_mat_indices(num_rows: int, num_cols: int, row_swaps: arr = (), col_swaps: arr = ()) -> tuple:
    if not (isinstance(num_rows, int) and num_rows > 0 and isinstance(num_cols, int) and num_cols > 0):
        raise ValueError("Number of rows and columns must be positive integers")
    if set(map(len, row_swaps)) != {2} or set(map(len, col_swaps)) != {2}:
        raise ValueError("Swap information must be tuples of two elements")
    row_coords = list(range(num_rows))
    col_coords = list(range(num_cols))

    def adjust_index(index, limit):
        return index + limit if index < 0 else index

    for r1, r2 in row_swaps:
        r1, r2 = adjust_index(r1, num_rows), adjust_index(r2, num_rows)
        if not (0 <= r1 < num_rows and 0 <= r2 < num_rows):
            raise ValueError("Row index out of range")
        row_coords[r1], row_coords[r2] = row_coords[r2], row_coords[r1]
    for c1, c2 in col_swaps:
        c1, c2 = adjust_index(c1, num_cols), adjust_index(c2, num_cols)
        if not (0 <= c1 < num_cols and 0 <= c2 < num_cols):
            raise ValueError("Column index out of range")
        col_coords[c1], col_coords[c2] = col_coords[c2], col_coords[c1]
    return row_coords, col_coords


def perm_mat(num_rows: int, num_cols: int, row_swaps: arr = (), col_swaps: arr = (), rtype: Callable = Matrix) -> Any:
    row_coords, col_coords = perm_mat_indices(num_rows, num_cols, row_swaps, col_swaps)
    result = [[0] * num_cols for _ in range(num_rows)]
    for i in range(num_rows):
        result[i][col_coords[row_coords[i]]] = 1
    return result if rtype is list else rtype(result)


def qr(matrix: Matrix, reduce: bool = False) -> tuple:
    from math import hypot
    r_ = matrix.rows
    c_ = matrix.cols
    q = identity(r_)
    r = matrix
    rows, cols = tril_indices(r_, -1, c_)
    for row, col in zip(rows, cols):
        if r[row][col] != 0:
            _r = hypot(r[col][col], r[row][col])
            _c = r[col][col] / _r
            _s = -r[row][col] / _r
            g = identity(r_)
            g[col][col] = _c
            g[row][row] = _c
            g[row][col] = _s
            g[col][row] = -_s
            q = q.inner(g)
            r = g @ r
    if reduce:
        minimum = min(r_, c_)
        q = Matrix(q[:, :minimum], False)
        r = Matrix(r[:minimum], False)
    return q, r


def hessenberg(matrix: Matrix) -> tuple:
    from math import sqrt
    a = matrix.copy()
    n = a.rows
    t = [0.0] * n
    if isinstance(a[-1, -1], int):
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
                a[i, k] *= scale_inv
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
                a[i, k] *= h
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
        return Matrix([[float(a[0, 0])]], False), Matrix([[1.0]], False)
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
    e = [[0] * n for _ in range(n)]
    for i in range(n):
        e[i][i] = a[i][i]
    e = Matrix(e, False)
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


def svd(matrix: Matrix, full: bool = False, calc_uv: bool = True) -> tuple:
    from math import hypot
    def svd_helper():
        w = [0.0] * n
        lb = [0.0] * n
        rb = [0.0] * n
        dw = [0.0] * n
        g = scale = anorm = 0
        for i in range(n):
            dw[i] = scale * g
            g = s = scale = 0
            if i < m:
                for k in range(i, m):
                    scale += abs(u[k, i].real) + abs(u[k, i].imag)
                if scale != 0:
                    for k in range(i, m):
                        u[k, i] /= scale
                        ar = u[k, i].real
                        ai = u[k, i].imag
                        s += ar * ar + ai * ai
                    f = u[i, i]
                    g = -s ** 0.5
                    if f.real < 0:
                        b = -g - f.conjugate()
                        g = -g
                    else:
                        b = -g + f.conjugate()
                    b /= b.conjugate()
                    b += 1
                    h = 2 * (f.real * g - s)
                    u[i, i] = f - g
                    b /= h
                    lb[i] = (b / scale) / scale
                    for j in range(i + 1, n):
                        s = 0
                        for k in range(i, m):
                            s += u[k, i].conjugate() * u[k, j]
                        f = b * s
                        for k in range(i, m):
                            u[k, j] += f * u[k, i]
                    for k in range(i, m):
                        u[k, i] *= scale
            sigma[i] = scale * g
            g = s = scale = 0
            if i < m and i != n - 1:
                for k in range(i + 1, n):
                    scale += abs(u[i, k].real) + abs(u[i, k].imag)
                if scale:
                    for k in range(i + 1, n):
                        u[i, k] /= scale
                        ar = u[i, k].real
                        ai = u[i, k].imag
                        s += ar * ar + ai * ai
                    f = u[i, i + 1]
                    g = -s ** 0.5
                    if f.real < 0:
                        b = -g - f.conjugate()
                        g = -g
                    else:
                        b = -g + f.conjugate()
                    b /= b.conjugate()
                    b += 1
                    h = 2 * (f.real * g - s)
                    u[i, i + 1] = f - g
                    b /= h
                    rb[i] = (b / scale) / scale
                    for k in range(i + 1, n):
                        w[k] = u[i, k]
                    for j in range(i + 1, m):
                        s = 0
                        for k in range(i + 1, n):
                            s += u[i, k].conjugate() * u[j, k]
                        f = s * b
                        for k in range(i + 1, n):
                            u[j, k] += f * w[k]
                    for k in range(i + 1, n):
                        u[i, k] *= scale
            anorm = max(anorm, abs(sigma[i]) + abs(dw[i]))
        if calc_uv:
            for i in range(n - 2, -1, -1):
                v[i + 1, i + 1] = 1
                if dw[i + 1] != 0:
                    f = rb[i].conjugate()
                    for j in range(i + 1, n):
                        v[i, j] = u[i, j] * f
                    for j in range(i + 1, n):
                        s = 0
                        for k in range(i + 1, n):
                            s += u[i, k].conjugate() * v[j, k]
                        for k in range(i + 1, n):
                            v[j, k] += s * v[i, k]
                for j in range(i + 1, n):
                    v[j, i] = v[i, j] = 0
            v[0, 0] = 1
            for i in range(min(m, n) - 1, -1, -1):
                g = sigma[i]
                for j in range(i + 1, n):
                    u[i, j] = 0
                if g != 0:
                    g = 1 / g
                    for j in range(i + 1, n):
                        s = 0
                        for k in range(i + 1, m):
                            s += u[k, i].conjugate() * u[k, j]
                        f = s * lb[i].conjugate()
                        for k in range(i, m):
                            u[k, j] += f * u[k, i]
                    for j in range(i, m):
                        u[j, i] *= g
                else:
                    for j in range(i, m):
                        u[j, i] = 0
                u[i, i] += 1
        for k in range(n - 1, -1, -1):
            its = 0
            while True:
                its += 1
                flag = True
                nm = None
                o = None
                for o in range(k, -1, -1):
                    nm = o - 1
                    if abs(dw[o]) + anorm == anorm:
                        flag = False
                        break
                    if abs(sigma[nm]) + anorm == anorm:
                        break
                if flag:
                    c = 0
                    s = 1
                    for i in range(o, k + 1):
                        f = s * dw[i]
                        dw[i] *= c
                        if abs(f) + anorm == anorm:
                            break
                        g = sigma[i]
                        h = hypot(f, g)
                        sigma[i] = h
                        h = 1 / h
                        c = g * h
                        s = -f * h
                        if calc_uv:
                            for j in range(m):
                                y = u[j, nm]
                                z = u[j, i]
                                u[j, nm] = y * c + z * s
                                u[j, i] = z * c - y * s
                z = sigma[k]
                if o == k:
                    if z < 0:
                        sigma[k] = -z
                        if calc_uv:
                            for j in range(n):
                                v[k, j] = -v[k, j]
                    break
                x = sigma[o]
                nm = k - 1
                y = sigma[nm]
                g = dw[nm]
                h = dw[k]
                f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y)
                g = hypot(f, 1)
                if f >= 0:
                    f = ((x - z) * (x + z) + h * ((y / (f + g)) - h)) / x
                else:
                    f = ((x - z) * (x + z) + h * ((y / (f - g)) - h)) / x
                c = s = 1
                for j in range(o, nm + 1):
                    g = dw[j + 1]
                    y = sigma[j + 1]
                    h = s * g
                    g = c * g
                    z = hypot(f, h)
                    dw[j] = z
                    c = f / z
                    s = h / z
                    f = x * c + g * s
                    g = g * c - x * s
                    h = y * s
                    y *= c
                    if calc_uv:
                        for jj in range(n):
                            x = v[j, jj]
                            z = v[j + 1, jj]
                            v[j, jj] = x * c + z * s
                            v[j + 1, jj] = z * c - x * s
                    z = hypot(f, h)
                    sigma[j] = z
                    if z != 0:
                        z = 1 / z
                        c = f * z
                        s = h * z
                    f = c * g + s * y
                    x = c * y - s * g
                    if calc_uv:
                        for jj in range(m):
                            y = u[jj, j]
                            z = u[jj, j + 1]
                            u[jj, j] = y * c + z * s
                            u[jj, j + 1] = z * c - y * s
                dw[o] = 0
                dw[k] = f
                sigma[k] = x
        for i in range(n):
            imax = i
            s = abs(sigma[i])
            for j in range(i + 1, n):
                c = abs(sigma[j])
                if c > s:
                    s = c
                    imax = j
            if imax != i:
                z = sigma[i]
                sigma[i] = sigma[imax]
                sigma[imax] = z
                if calc_uv:
                    for j in range(m):
                        z = u[j, i]
                        u[j, i] = u[j, imax]
                        u[j, imax] = z
                    for j in range(n):
                        z = v[i, j]
                        v[i, j] = v[imax, j]
                        v[imax, j] = z

    m, n = matrix.rows, matrix.cols
    sigma = [0.0] * max(m, n) if full else [0.0] * n
    u = matrix.copy()
    if not calc_uv:
        svd_helper()
        sigma = sigma[:min(m, n)]
        return diag(sigma),
    elif full and n < m:
        v = Matrix([[0] * m for _ in range(m)], False)
        d = m - n
        u = Matrix([_ + [0] * d for _ in u.data], False)
        _, n = n, m
        svd_helper()
        sigma = sigma[:_]
        v = v[:_, :_]
        return u, diag(sigma), Matrix(v, False)
    else:
        v = Matrix([[0] * n for _ in range(n)], False)
        svd_helper()
        if n > m:
            if not full:
                v = Matrix(v[:m, :], False)
            sigma = sigma[:m]
            u = Matrix(u[:, :m], False)
        return u, diag(sigma), v


del Array
