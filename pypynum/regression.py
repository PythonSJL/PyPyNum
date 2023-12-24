def linear_regression(x_values, y_values):
    from .mathematics import mean
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    numerator = sum([(x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)])
    denominator = sum([(x - x_mean) ** 2 for x in x_values])
    beta = numerator / denominator
    alpha = y_mean - beta * x_mean
    print("f(x) = {} * x + {}".format(round(beta, 9), round(alpha, 9)))
    return [round(beta, 9), round(alpha, 9)]


def __covariance_matrix(_x, _y):
    from .mathematics import var
    x2 = [_ ** 2 for _ in _x]
    xy = [round(a + b, 12) for a, b in zip(_x, _y)]
    x2x = [round(a ** 2 + a, 12) for a in _x]
    x2y = [round(a ** 2 + b, 12) for a, b in zip(_x, _y)]
    vx2 = var(x2)
    vxy = var(xy)
    vx2x = var(x2x)
    vx2y = var(x2y)
    v1y = (vxy - var(_x) - var(_y)) / 2
    v2y = (vx2y - vx2 - var(_y)) / 2
    v12 = (vx2x - vx2 - var(_x)) / 2
    return [var(_x), v12, v1y, v12, vx2, v2y]


def __solve_equations(a1, b1, c1, a2, b2, c2):
    denominator = a1 * b2 - a2 * b1
    if denominator == 0:
        return None
    x = (c1 * b2 - c2 * b1) / denominator
    y = (a1 * c2 - a2 * c1) / denominator
    return [x, y]


def parabolic_regression(x, y):
    from .mathematics import mean
    o = __covariance_matrix(x, y)
    b, a = __solve_equations(o[0], o[1], o[2], o[3], o[4], o[5])
    c = mean(y) - b * mean(x) - a * mean([_ ** 2 for _ in x])
    print("f(x) = {} * x ** 2 + {} * x + {}".format(round(a, 9), round(b, 9), round(c, 9)))
    return [round(a, 9), round(b, 9), round(c, 9)]
