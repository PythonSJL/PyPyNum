from .kernels import matmul2x2kernel, matmul3x3kernel, matmul4x4kernel
from .tools import fast_pow
from .types import Union, arr, real


def pascal(n: int) -> list:
    triangle = [[1]]
    for i in range(1, n + 1):
        new_row = [1]
        for j in range(1, i):
            new_row.append(triangle[i - 1][j - 1] + triangle[i - 1][j])
        new_row.append(1)
        triangle.append(new_row)
    return triangle


def bell(n: int) -> list:
    triangle = [[1]]
    for i in range(1, n + 1):
        new_row = [triangle[i - 1][-1]]
        for j in range(1, i + 1):
            new_row.append(new_row[j - 1] + triangle[i - 1][j - 1])
        triangle.append(new_row)
    return triangle


def stirling1(n: int) -> list:
    triangle = [[1]]
    for i in range(1, n + 1):
        new_row = [0]
        for j in range(1, i):
            new_row.append((i - 1) * triangle[i - 1][j] + triangle[i - 1][j - 1])
        new_row.append(triangle[i - 1][i - 1])
        triangle.append(new_row)
    return triangle


def stirling2(n: int) -> list:
    triangle = [[1]]
    for i in range(1, n + 1):
        new_row = [0]
        for j in range(1, i):
            new_row.append(j * triangle[i - 1][j] + triangle[i - 1][j - 1])
        new_row.append(triangle[i - 1][i - 1])
        triangle.append(new_row)
    return triangle


def farey(n: int) -> list:
    a2, b2 = None, None
    if n == 0:
        return []
    elif n == 1:
        return [(0, 1), (1, 1)]
    else:
        f = farey(n - 1)
        m = len(f) - 1
        result = []
        for i in range(m):
            a1, b1 = f[i]
            a2, b2 = f[i + 1]
            result.append((a1, b1))
            temp = (a1 + a2, b1 + b2)
            if temp[0] <= temp[1] <= n:
                result.append(temp)
        result.append((a2, b2))
        return result


def fibonacci(n: int, single: bool = True) -> Union[int, list]:
    if isinstance(n, float):
        n = int(n)
    if not single:
        if n == 0:
            return [0]
        res = [0, 1]
        for i in range(2, n + 1):
            res.append(res[i - 1] + res[i - 2])
        return res
    if n <= 1:
        return max(n, 0)
    a = [[1, 1], [1, 0]]
    res = fast_pow(a, n, [[1, 0], [0, 1]], matmul2x2kernel)
    return res[0][1]


def pell(n: int, single: bool = True) -> Union[int, list]:
    if isinstance(n, float):
        n = int(n)
    if not single:
        if n == 0:
            return [0]
        res = [0, 1]
        for i in range(2, n + 1):
            res.append(2 * res[i - 1] + res[i - 2])
        return res
    if n <= 1:
        return max(n, 0)
    a = [[2, 1], [1, 0]]
    res = fast_pow(a, n, [[1, 0], [0, 1]], matmul2x2kernel)
    return res[0][1]


def lucas(n: int, single: bool = True) -> Union[int, list]:
    if isinstance(n, float):
        n = int(n)
    if not single:
        if n == 0:
            return [2]
        res = [2, 1]
        for i in range(2, n + 1):
            res.append(res[i - 1] + res[i - 2])
        return res
    if n == 0:
        return 2
    if n == 1:
        return 1
    a = [[1, 1], [1, 0]]
    res = fast_pow(a, n, [[1, 0], [0, 1]], matmul2x2kernel)
    return res[0][0] + res[1][1]


def tribonacci(n: int, single: bool = True) -> Union[int, list]:
    if isinstance(n, float):
        n = int(n)
    res = [0, 1, 1]
    if not single:
        if n < 3:
            return res[:n]
        for i in range(3, n + 1):
            res.append(res[i - 1] + res[i - 2] + res[i - 3])
        return res[:n] if n < 3 else res
    if n < 3:
        return res[n]
    a = [[1, 1, 1], [1, 0, 0], [0, 1, 0]]
    res = fast_pow(a, n - 1, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], matmul3x3kernel)
    return res[0][0]


def tetranacci(n: int, single: bool = True) -> Union[int, list]:
    if isinstance(n, float):
        n = int(n)
    res = [0, 1, 1, 2]
    if not single:
        if n < 4:
            return res[:n]
        for i in range(4, n + 1):
            res.append(res[i - 1] + res[i - 2] + res[i - 3] + res[i - 4])
        return res
    if n < 4:
        return res[n]
    a = [[1, 1, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    res = fast_pow(a, n - 1, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], matmul4x4kernel)
    return res[0][0]


def catalan(n: int, single: bool = True) -> Union[int, list]:
    try:
        from math import comb
    except ImportError:
        from .maths import combination as comb
    if isinstance(n, float):
        n = int(n)
    if not single:
        if n == 0:
            return [1]
        res = [1]
        for i in range(1, n + 1):
            res.append(res[i - 1] * (4 * i - 2) // (i + 1))
        return res
    return comb(2 * n, n) // (n + 1)


def bernoulli(n: int, single: bool = True) -> arr:
    from fractions import Fraction
    try:
        from math import comb
    except ImportError:
        from .maths import combination as comb
    if isinstance(n, float):
        n = int(n)
    result = [1, Fraction(-0.5)]
    if not single and n == 0:
        return [(1, 1)]
    for m in range(2, n + 1):
        if m & 1:
            b = 0
        else:
            s = sum([comb(m + 1, i) * result[i] for i in range(m)])
            b = -s / (m + 1)
        result.append(b)
    return result[n].as_integer_ratio() if single else [n.as_integer_ratio() for n in result]


def recaman(n: int, single: bool = True) -> Union[int, list]:
    if isinstance(n, float):
        n = int(n)
    seq = []
    for i in range(n + 1):
        if i == 0:
            x = 0
        else:
            x = seq[i - 1] - i
        if x >= 0 and x not in seq:
            seq += [x]
        else:
            seq += [seq[i - 1] + i]
    return seq[n] if single else seq


def sylvester(n: int, single: bool = True) -> Union[int, list]:
    if isinstance(n, float):
        n = int(n)
    result = [2]
    a = 2
    for _ in range(n):
        a = a * a - a + 1
        result.append(a)
    return result[n] if single else result


def padovan(n: int, single: bool = True) -> Union[int, list]:
    if isinstance(n, float):
        n = int(n)
    result = [1, 0, 0]
    if not single and n < len(result):
        return result[:n + 1]
    for i in range(3, n + 1):
        result.append(result[i - 2] + result[i - 3])
    return result[n] if single else result


def perrin(n: int, single: bool = True) -> Union[int, list]:
    if isinstance(n, float):
        n = int(n)
    result = [3, 0, 2]
    if not single and n < len(result):
        return result[:n + 1]
    for i in range(3, n + 1):
        result.append(result[i - 2] + result[i - 3])
    return result[n] if single else result


def pelllucas(n: int, single: bool = True) -> Union[int, list]:
    if isinstance(n, float):
        n = int(n)
    if not single:
        if n == 0:
            return [2]
        res = [2, 2]
        for i in range(2, n + 1):
            res.append(2 * res[i - 1] + res[i - 2])
        return res
    if n <= 1:
        return 2
    a = [[2, 1], [1, 0]]
    res = fast_pow(a, n, [[1, 0], [0, 1]], matmul2x2kernel)
    return res[0][0] + res[1][1]


def arithmetic_sequence(*, a1: real = None, an: real = None, d: real = None, n: real = None, s: real = None) -> dict:
    if [a1, an, d, n, s].count(None) != 2:
        raise ValueError("Must provide exactly three valid parameters")

    def solve_quadratic_equation(a, b, c):
        delta = b ** 2 - 4 * a * c
        if delta > 0:
            x1 = (-b + delta ** 0.5) / (2 * a)
            x2 = (-b - delta ** 0.5) / (2 * a)
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


def geometric_sequence(*, a1: real = None, an: real = None, r: real = None, n: real = None, s: real = None) -> dict:
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
        a1 = s * (1 - r) + an * r
        n = log(an / a1, r) + 1
    elif an is not None and n is not None and s is not None:
        a1 = NotImplemented
        r = NotImplemented
    elif r is not None and n is not None and s is not None:
        a1 = s * (r - 1) / (r ** n - 1)
        an = a1 * r ** (n - 1)
    if isinstance(n, list):
        n = sorted(n)
    return {"a1": a1, "an": an, "r": r, "n": n, "s": s}
