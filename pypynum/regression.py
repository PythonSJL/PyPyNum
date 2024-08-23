from .types import arr


def lin_reg(x: arr, y: arr) -> list:
    from .maths import mean
    if len(x) != len(y):
        raise ValueError("The array length of the independent and dependent variables must be equal")
    x_mean = mean(x)
    y_mean = mean(y)
    numerator = sum([(x - x_mean) * (y - y_mean) for x, y in zip(x, y)])
    denominator = sum([(x - x_mean) ** 2 for x in x])
    alpha = numerator / denominator
    beta = y_mean - alpha * x_mean
    return [alpha, beta]


def par_reg(x: arr, y: arr) -> list:
    from .maths import mean, var
    if len(x) != len(y):
        raise ValueError("The array length of the independent and dependent variables must be equal")

    def solve_equations(a1, b1, c1, a2, b2, c2):
        denominator = a1 * b2 - a2 * b1
        if denominator == 0:
            return None
        return [(c1 * b2 - c2 * b1) / denominator, (a1 * c2 - a2 * c1) / denominator]

    x2 = [_ ** 2 for _ in x]
    xy = [a + b for a, b in zip(x, y)]
    x2x = [a ** 2 + a for a in x]
    x2y = [a ** 2 + b for a, b in zip(x, y)]
    vx2 = var(x2)
    vxy = var(xy)
    vx2x = var(x2x)
    vx2y = var(x2y)
    v1y = (vxy - var(x) - var(y)) / 2
    v2y = (vx2y - vx2 - var(y)) / 2
    v12 = (vx2x - vx2 - var(x)) / 2
    o = [var(x), v12, v1y, v12, vx2, v2y]
    b, a = solve_equations(o[0], o[1], o[2], o[3], o[4], o[5])
    c = mean(y) - b * mean(x) - a * mean([_ ** 2 for _ in x])
    return [a, b, c]


def poly_reg(x: arr, y: arr, n: int = None) -> list:
    from .Matrix import mat
    if len(x) != len(y):
        raise ValueError("The array length of the independent and dependent variables must be equal")
    if n is None:
        n = len(x) - 1
    if not isinstance(n, int) or n <= 0:
        raise ValueError("The degree of a polynomial must be a natural number")
    m = len(x)
    _x = [[_ ** (n - i) for _ in x] for i in range(n)]
    _x.append([1] * m)
    _x = mat(_x)
    return ((_x @ _x.t()).inv() @ _x @ mat([list(y)]).t()).flatten()
