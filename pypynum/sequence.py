def fibonacci(n: int) -> int:
    if isinstance(n, float):
        n = int(n)

    def mul(_a, _b):
        c = [[0, 0], [0, 0]]
        c[0][0] = _a[0][0] * _b[0][0] + _a[0][1] * _b[1][0]
        c[0][1] = _a[0][0] * _b[0][1] + _a[0][1] * _b[1][1]
        c[1][0] = _a[1][0] * _b[0][0] + _a[1][1] * _b[1][0]
        c[1][1] = _a[1][0] * _b[0][1] + _a[1][1] * _b[1][1]
        return c

    if n <= 1:
        return max(n, 0)
    res = [[1, 0], [0, 1]]
    a = [[1, 1], [1, 0]]
    while n:
        if n & 1:
            res = mul(res, a)
        a = mul(a, a)
        n >>= 1
    return res[0][1]


def catalan(n: int) -> int:
    try:
        from math import comb
    except ImportError:
        from .maths import combination as comb
    return comb(2 * n, n) // (n + 1)


def bernoulli(n: int, single: bool = True) -> list:
    from fractions import Fraction
    try:
        from math import comb
    except ImportError:
        from .maths import combination as comb
    if isinstance(n, float):
        n = int(n)
    result = [1, Fraction(-0.5)]
    for m in range(2, n + 1):
        if m & 1:
            b = 0
        else:
            s = sum([comb(m + 1, i) * result[i] for i in range(m)])
            b = -s / (m + 1)
        result.append(b)
    return [result[n].numerator, result[n].denominator] if single else [[n.numerator, n.denominator] for n in result]
