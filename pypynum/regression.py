def linear_regression(x, y):
    from .mathematics import mean
    x_mean = mean(x)
    y_mean = mean(y)
    numerator = sum([(x - x_mean) * (y - y_mean) for x, y in zip(x, y)])
    denominator = sum([(x - x_mean) ** 2 for x in x])
    alpha = numerator / denominator
    beta = y_mean - alpha * x_mean
    print("f(x) = {} * x + {}".format(round(alpha, 9), round(beta, 9)))
    return [round(alpha, 9), round(beta, 9)]


def parabolic_regression(x, y):
    def solve_equations(a1, b1, c1, a2, b2, c2):
        denominator = a1 * b2 - a2 * b1
        if denominator == 0:
            return None
        return [(c1 * b2 - c2 * b1) / denominator, (a1 * c2 - a2 * c1) / denominator]

    from .mathematics import mean, var
    x2 = [_ ** 2 for _ in x]
    xy = [round(a + b, 12) for a, b in zip(x, y)]
    x2x = [round(a ** 2 + a, 12) for a in x]
    x2y = [round(a ** 2 + b, 12) for a, b in zip(x, y)]
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
    print("f(x) = {} * x ** 2 + {} * x + {}".format(round(a, 9), round(b, 9), round(c, 9)))
    return [round(a, 9), round(b, 9), round(c, 9)]
