class Matrix:
    def __init__(self, data):
        self.rows = len(data)
        self.cols = len(data[0])
        self.data = data

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions do not match")
        return Matrix([[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])

    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions do not match")
        return Matrix([[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrix dimensions do not match")
            return Matrix([[self.data[i][j] * other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])
        else:
            return Matrix([[self.data[b][a] * other for a in range(self.cols)] for b in range(self.rows)])

    def __matmul__(self, other):
        if self.cols != other.rows:
            raise ValueError("Matrix dimensions do not match")
        result = [[0 for _ in range(other.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return Matrix(result)

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
            raise ValueError("Matrix must be square")
        return sum([self.data[i][i] for i in range(self.rows)])

    def det(self):
        if self.rows != self.cols:
            raise ValueError("Matrix must be square")
        matrix = [[self.data[b][a] for a in range(self.cols)] for b in range(self.rows)]
        size = len(matrix)
        determinant = 1
        for i in range(size):
            if matrix[i][i] == 0:
                for j in range(i + 1, size):
                    if matrix[i][j] != 0:
                        matrix[i], matrix[j] = matrix[j], matrix[i]
                        determinant = -determinant
                        break
                else:
                    return 0
            determinant *= matrix[i][i]
            for j in range(i + 1, size):
                matrix[i][j] /= matrix[i][i]
                for k in range(i + 1, size):
                    matrix[k][j] -= matrix[i][j] * matrix[k][i]
        return determinant

    def inv(self):
        if self.rows != self.cols:
            raise ValueError("Matrix must be square")
        det = self.det()
        if det == 0:
            return None
        matrix_minor = [[Matrix(Matrix(self.data).minor(i, j).data).det() for j in range(self.cols)] for i in
                        range(self.rows)]
        cofactors = [[x * (-1) ** (row + col) for col, x in enumerate(matrix_minor[row])] for row in range(self.cols)]
        adjugate = Matrix(cofactors).t().data
        return Matrix(adjugate) * (1 / det)

    def simplest(self):
        matrix = [[self.data[b][a] for a in range(self.cols)] for b in range(self.rows)]
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
        matrix = self.simplest().data
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
            raise ValueError("Matrix must be square")
        result = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.cols):
            result[i][i] = 1
        return Matrix(result)

    def __str__(self):
        return str(self.data).replace("], ", "]\n").replace(",", "")

    def str(self):
        _max = len(max(str(self.data).replace("[", "").replace("]", "").replace(",", "").split(), key=len))
        _then = str(_format(self.data, _max)).replace("], ", "]\n").replace(",", "").replace("'", "").split("\n")
        _max = max([_.count("[") for _ in _then])
        return "\n".join([(_max - _.count("[")) * " " + _ + "\n" * (_.count("]") - 1) for _ in _then]).strip()


def _format(_nested_list, _max_length):
    if isinstance(_nested_list, list):
        _copy = []
        for item in _nested_list:
            _copy.append(_format(item, _max_length))
        return _copy
    else:
        _item = str(_nested_list)
        return " " * (_max_length - len(_item)) + _item


def _zeros(_dimensions):
    if len(_dimensions) == 0:
        return 0
    else:
        _matrix = []
        for i in range(_dimensions[0]):
            _row = _zeros(_dimensions[1:])
            _matrix.append(_row)
        return _matrix


def zeros(_dimensions):
    return Matrix(_zeros(_dimensions))


def _zeros_like(_nested_list):
    if isinstance(_nested_list, list):
        _copy = []
        for item in _nested_list:
            _copy.append(_zeros_like(item))
        return _copy
    else:
        return 0


def zeros_like(_nested_list):
    return Matrix(_zeros_like(_nested_list))
