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


def arithmetic_sequence(*, a1=None, an=None, d=None, n=None, s=None):
    if [a1, an, d, n, s].count(None) != 2:
        raise ValueError("Must provide exactly three valid parameters")

    def solve_quadratic_equation(a, b, c):
        delta = b ** 2 - 4 * a * c
        if delta > 0:
            x1 = (-b + (delta ** 0.5)) / (2 * a)
            x2 = (-b - (delta ** 0.5)) / (2 * a)
            return [x1, x2]
        elif delta == 0:
            x = -b / (2 * a)
            return [x]
        else:
            return []

    if any([not isinstance(val, (int, float, type(None))) for val in [a1, an, d, n, s]]):
        raise ValueError("The input parameters must all be real numbers")
    if a1 is not None and an is not None and d is not None:
        n = (an - a1) / d + 1
        s = n / 2 * (2 * a1 + (n - 1) * d)
    elif a1 is not None and an is not None and n is not None:
        d = (an - a1) / (n - 1)
        s = n / 2 * (a1 + an)
    elif a1 is not None and an is not None and s is not None:
        n = s * 2 / (a1 + an)
        d = (an - a1) / (n - 1)
    elif a1 is not None and d is not None and n is not None:
        an = a1 + (n - 1) * d
        s = n / 2 * (2 * a1 + (n - 1) * d)
    elif a1 is not None and d is not None and s is not None:
        n = [item for item in solve_quadratic_equation(d, 2 * a1 - d, -2 * s) if item > 0]
        if len(n) == 1:
            n = n[0]
            an = a1 + (n - 1) * d
    elif a1 is not None and n is not None and s is not None:
        an = 2 * s / n - a1
        d = (an - a1) / (n - 1)
    elif an is not None and d is not None and n is not None:
        a1 = an - (n - 1) * d
        s = n / 2 * (a1 + an)
    elif an is not None and d is not None and s is not None:
        n = [item for item in solve_quadratic_equation(-d, 2 * an + d, -2 * s) if item > 0]
        if len(n) == 1:
            n = n[0]
            a1 = an - (n - 1) * d
    elif an is not None and n is not None and s is not None:
        d = (an * n - s) / (n * (n - 1) / 2)
        a1 = an - (n - 1) * d
    elif d is not None and n is not None and s is not None:
        a1 = (s / n) - (d * (n - 1)) / 2
        an = a1 + (n - 1) * d
    if isinstance(n, list):
        n = sorted(n)
    return {"a1": a1, "an": an, "d": d, "n": n, "s": s}


def geometric_sequence(*, a1=None, an=None, r=None, n=None, s=None):
    from math import log
    if [a1, an, r, n, s].count(None) != 2:
        raise ValueError("Must provide exactly three valid parameters")
    if any([not isinstance(val, (int, float, type(None))) for val in [a1, an, r, n, s]]):
        raise ValueError("The input parameters must all be real numbers")
    if a1 is not None and an is not None and r is not None:
        n = log(an / a1, r) + 1
        s = a1 * (r ** n - 1) / (r - 1) if r != 1 else a1 * n
    elif a1 is not None and an is not None and n is not None:
        r = (an / a1) ** (1 / (n - 1))
        s = a1 * (r ** n - 1) / (r - 1) if r != 1 else a1 * n
    elif a1 is not None and an is not None and s is not None:
        r = (s - a1) / (s - an)
        n = log(an / a1, r) + 1
    elif a1 is not None and r is not None and n is not None:
        an = a1 * r ** (n - 1)
        s = a1 * (r ** n - 1) / (r - 1) if r != 1 else a1 * n
    elif a1 is not None and r is not None and s is not None:
        an = (s * r - s + a1) / r
        n = log(an / a1, r) + 1
    elif a1 is not None and n is not None and s is not None:
        an = NotImplemented
        r = NotImplemented
    elif an is not None and r is not None and n is not None:
        a1 = an / r ** (n - 1)
        s = a1 * (r ** n - 1) / (r - 1) if r != 1 else a1 * n
    elif an is not None and r is not None and s is not None:
        a1 = NotImplemented
        n = NotImplemented
    elif an is not None and n is not None and s is not None:
        a1 = NotImplemented
        r = NotImplemented
    elif r is not None and n is not None and s is not None:
        a1 = s * (r - 1) / (r ** n - 1)
        an = a1 * r ** (n - 1)
    if isinstance(n, list):
        n = sorted(n)
    return {"a1": a1, "an": an, "r": r, "n": n, "s": s}
