import math
from .types import arr, num, real


def root(x: num, y: num) -> num:
    """
    introduction
    ==========
    Root \n
    ʸ√x̄

    example
    ==========
    >>> root(2, 4)
    1.189207115002721
    >>>
    :param x: integer | float | complex
    :param y: integer | float | complex
    :return:
    """
    flag = False
    if x < 0 and int(y) == y and int(y) & 1:
        x = -x
        flag = True
    return -x ** (1 / y) if flag else x ** (1 / y)


def exp(x: real) -> real:
    """
    introduction
    ==========
    Exponential \n
    eˣ

    example
    ==========
    >>> exp(0.5)
    1.6487212707001282
    >>>
    :param x: int | float
    :return:
    """
    return math.exp(x)


def ln(x: real) -> real:
    """
    introduction
    ==========
    Natural logarithm \n
    logₑx

    example
    ==========
    >>> ln(0.5)
    -0.6931471805599453
    >>>
    :param x: int | float
    :return:
    """
    return math.log(x)


def gcd(*args: int) -> int:
    """
    introduction
    ==========
    The greatest common factor of multiple integers

    example
    ==========
    >>> gcd(12, 18, 24)
    6
    >>>
    :param args: integer
    :return:
    """
    if not args:
        return 0
    gcd_value = args[0]
    for n in args[1:]:
        gcd_value = math.gcd(gcd_value, n)
    return gcd_value


def lcm(*args: int) -> int:
    """
    introduction
    ==========
    The least common multiple of multiple integers

    example
    ==========
    >>> lcm(12, 18, 24)
    72
    >>>
    :param args: integer
    :return:
    """
    if not args:
        return 0
    lcm_value = args[0]
    for n in args[1:]:
        lcm_value = math.lcm(lcm_value, n)
    return lcm_value


def sin(x: real) -> real:
    """
    Sine
    :param x: int | float
    :return:
    """
    return math.sin(x)


def cos(x: real) -> real:
    """
    Cosine
    :param x: int | float
    :return:
    """
    return math.cos(x)


def tan(x: real) -> real:
    """
    Tangent
    :param x: int | float
    :return:
    """
    return math.tan(x)


def csc(x: real) -> real:
    """
    Cosecant
    :param x: int | float
    :return:
    """
    return 1 / sin(x)


def sec(x: real) -> real:
    """
    Secant
    :param x: int | float
    :return:
    """
    return 1 / cos(x)


def cot(x: real) -> real:
    """
    Cotangent
    :param x: int | float
    :return:
    """
    return cos(x) / sin(x)


def asin(x: real) -> real:
    """
    Arcsine
    :param x: int | float
    :return:
    """
    return math.asin(x)


def acos(x: real) -> real:
    """
    Arccosine
    :param x: int | float
    :return:
    """
    return math.acos(x)


def atan(x: real) -> real:
    """
    Arctangent
    :param x: int | float
    :return:
    """
    return math.atan(x)


def acsc(x: real) -> real:
    """
    Arccosecant
    :param x: int | float
    :return:
    """
    return asin(1 / x)


def asec(x: real) -> real:
    """
    Arcsecant
    :param x: int | float
    :return:
    """
    return acos(1 / x)


def acot(x: real) -> real:
    """
    Arccotangent
    :param x: int | float
    :return:
    """
    return atan(1 / x)


def sinh(x: real) -> real:
    """
    Hyperbolic Sine
    :param x: int | float
    :return:
    """
    return math.sinh(x)


def cosh(x: real) -> real:
    """
    Hyperbolic Cosine
    :param x: int | float
    :return:
    """
    return math.cosh(x)


def tanh(x: real) -> real:
    """
    Hyperbolic Tangent
    :param x: int | float
    :return:
    """
    return math.tanh(x)


def csch(x: real) -> real:
    """
    Hyperbolic Cosecant
    :param x: int | float
    :return:
    """
    return 1 / sinh(x)


def sech(x: real) -> real:
    """
    Hyperbolic Secant
    :param x: int | float
    :return:
    """
    return 1 / cosh(x)


def coth(x: real) -> real:
    """
    Hyperbolic Cotangent
    :param x: int | float
    :return:
    """
    return cosh(x) / sinh(x)


def asinh(x: real) -> real:
    """
    Hyperbolic Arcsine
    :param x: int | float
    :return:
    """
    return math.asinh(x)


def acosh(x: real) -> real:
    """
    Hyperbolic Arccosine
    :param x: int | float
    :return:
    """
    return math.acosh(x)


def atanh(x: real) -> real:
    """
    Hyperbolic Arctangent
    :param x: int | float
    :return:
    """
    return math.atanh(x)


def acsch(x: real) -> real:
    """
    Hyperbolic Arccosecant
    :param x: int | float
    :return:
    """
    return asinh(1 / x)


def asech(x: real) -> real:
    """
    Hyperbolic Arcsecant
    :param x: int | float
    :return:
    """
    return acosh(1 / x)


def acoth(x: real) -> real:
    """
    Hyperbolic Arccotangent
    :param x: int | float
    :return:
    """
    return atanh(1 / x)


def ptp(numbers: arr) -> num:
    """
    introduction
    ==========
    Range of numbers

    example
    ==========
    >>> ptp([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    9
    >>>
    :param numbers: list | tuple
    :return:
    """
    return max(numbers) - min(numbers)


def median(numbers: arr) -> num:
    """
    introduction
    ==========
    Median of numbers

    example
    ==========
    >>> median([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    5.5
    >>>
    :param numbers: list | tuple
    :return:
    """
    data = sorted(numbers)
    n = len(data)
    if n % 2 == 0:
        return (data[n // 2 - 1] + data[n // 2]) / 2
    else:
        return data[n // 2]


def freq(data: arr) -> dict:
    """
    introduction
    ==========
    Frequency of data

    example
    ==========
    >>> freq([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}
    >>>
    :param data: list | tuple
    :return:
    """
    result = {}
    for d in data:
        if d in result:
            result[d] += 1
        else:
            result[d] = 1
    return result


def mode(data: arr):
    """
    introduction
    ==========
    Mode of data

    example
    ==========
    >>> mode([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    1
    >>>
    :param data: list | tuple
    :return:
    """
    count_dict = freq(data)
    max_count = 0
    max_num = 0
    for k, v in count_dict.items():
        if v > max_count:
            max_count = v
            max_num = k
    return max_num


def mean(numbers: arr) -> num:
    """
    introduction
    ==========
    Average of numbers

    example
    ==========
    >>> mean([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    5.5
    >>>
    :param numbers: list | tuple
    :return:
    """
    return sum(numbers) / len(numbers)


def var(numbers: arr) -> num:
    """
    introduction
    ==========
    Variance of numbers

    example
    ==========
    >>> var([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    8.25
    >>>
    :param numbers: list | tuple
    :return:
    """
    avg = mean(numbers)
    return sum([(x - avg) ** 2 for x in numbers]) / len(numbers)


def std(numbers: arr) -> num:
    """
    introduction
    ==========
    Standard deviation of numbers

    example
    ==========
    >>> std([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    2.8722813232690143
    >>>
    :param numbers: list | tuple
    :return:
    """
    return var(numbers) ** 0.5


def product(numbers: arr) -> num:
    """
    introduction
    ==========
    Product of multiple numbers

    example
    ==========
    >>> product([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    3628800
    >>>
    :param numbers: list | tuple
    :return:
    """
    result = 1
    for _ in numbers:
        result *= _
    return result


def sigma(i: int, n: int, f) -> num:
    """
    introduction
    ==========
    n \n
    Σ  f(x) \n
    i

    example
    ==========
    >>> sigma(1, 10, lambda x: x ** x)
    10405071317
    >>>
    :param i: integer
    :param n: integer
    :param f: function
    :return:
    """
    return sum([f(_) for _ in range(i, n + 1)])


def pi(i: int, n: int, f) -> num:
    """
    introduction
    ==========
    n \n
    Π  f(x) \n
    i

    example
    ==========
    >>> pi(1, 10, lambda x: x ** x)
    215779412229418562091680268288000000000000000
    >>>
    :param i: integer
    :param n: integer
    :param f: function
    :return:
    """
    return product([f(_) for _ in range(i, n + 1)])


def derivative(f, x: real, h: real = 1e-7) -> float:
    """
    introduction
    ==========
    Derivative calculation

    example
    ==========
    >>> derivative(lambda a: a ** 2, 2)
    3.9999999956741306
    >>>
    :param f: function
    :param x: integer | float
    :param h: integer | float
    :return:
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def definite_integral(f, x_start: real, x_end: real, n: int = 10000000) -> float:
    """
    introduction
    ==========
    Definite integral calculation

    example
    ==========
    >>> definite_integral(lambda a: a ** 2, 0, 2)
    2.6666666666664036
    >>>
    :param f: function
    :param x_start: integer | float
    :param x_end: integer | float
    :param n: integer | float
    :return:
    """
    h = (x_end - x_start) / n
    _sum = 0
    for i in range(n):
        x = x_start + i * h
        _sum += f(x + h) + f(x)
    return _sum * h / 2


def beta(p: real, q: real) -> real:
    """
    introduction
    ==========
    Beta function calculation \n
    Β(p, q)

    example
    ==========
    >>> beta(0.5, 0.5)
    3.1415926535897927
    >>>
    :param p: integer | float
    :param q: integer | float
    :return:
    """
    return gamma(p) * gamma(q) / gamma(p + q)


def gamma(alpha: real) -> float:
    """
    introduction
    ==========
    Gamma function calculation \n
    Γ(s)

    example
    ==========
    >>> gamma(1.5)
    0.886226925452758
    >>>
    :param alpha: integer | float
    :return:
    """
    return math.gamma(alpha)


def factorial(n: int) -> int:
    """
    introduction
    ==========
    Integer factorial

    example
    ==========
    >>> factorial(10)
    3628800
    >>>
    :param n: integer
    :return:
    """
    return math.factorial(n)


def arrangement(n: int, r: int) -> int:
    """
    introduction
    ==========
    A(n, r)

    example
    ==========
    >>> arrangement(10, 5)
    30240
    >>>
    :param n: integer
    :param r: integer
    :return:
    """
    return product(list(range(n, n - r, -1)))


def combination(n: int, r: int) -> int:
    """
    introduction
    ==========
    C(n, r)

    example
    ==========
    >>> combination(10, 5)
    252
    >>>
    :param n: integer
    :param r: integer
    :return:
    """
    return product(list(range(n, n - r, -1))) // product(list(range(r, 0, -1)))


def zeta(alpha: real) -> float:
    """
    introduction
    ==========
    Zeta function calculation \n
    ζ(s)

    example
    ==========
    >>> zeta(2)
    1.6449339668472496
    >>>
    :param alpha: integer | float
    :return:
    """
    return sum([1 / _ ** alpha for _ in range(1, 10000000)])


def gaussian(x: real, _mu: real = 0, _sigma: real = 1) -> float:
    """
    introduction
    ==========
    Gaussian function calculation

    example
    ==========
    >>> gaussian(0)
    0.3989422804014327
    >>>
    :param x: integer | float
    :param _mu: integer | float
    :param _sigma: integer | float
    :return:
    """
    return exp(-(x - _mu) ** 2 / (2 * _sigma ** 2)) / (_sigma * 2.5066282746310002)


def poisson(x: int, _lambda: real) -> float:
    """
    introduction
    ==========
    Poisson distribution calculation

    example
    ==========
    >>> poisson(1, 1)
    0.36787944117144233
    >>>
    :param x: integer
    :param _lambda: integer | float
    :return:
    """
    if isinstance(x, float):
        x = int(x)
    return _lambda ** x * 2.718281828459045 ** (-_lambda) / factorial(x)


def erf(x: real) -> float:
    """
    introduction
    ==========
    Error function calculation

    example
    ==========
    >>> erf(1)
    0.842700792949715
    >>>
    :param x: integer | float
    :return:
    """
    return math.erf(x)


def sigmoid(x: real) -> float:
    """
    introduction
    ==========
    Sigmoid function calculation

    example
    ==========
    >>> sigmoid(1)
    0.7310585786300049
    >>>
    :param x: integer | float
    :return:
    """
    return 1 / (1 + exp(-x))


def sign(x: real) -> int:
    """
    introduction
    ==========
    Sign function calculation

    example
    ==========
    >>> sign(10)
    1
    >>>
    :param x: integer | float
    :return:
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def parity(x: int) -> int:
    """
    introduction
    ==========
    Calculate (-1) ** n using bits

    example
    ==========
    >>> parity(10)
    1
    >>>
    :param x: integer
    :return:
    """
    return (~x & 1) * 2 - 1


def cumsum(lst: arr) -> list:
    """
    introduction
    ==========
    Sequence accumulation

    example
    ==========
    >>> cumsum([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]
    >>>
    :param lst: list | tuple
    :return:
    """
    if not lst:
        return []
    cumulative_sum = [lst[0]]
    for i in range(1, len(lst)):
        cumulative_sum.append(cumulative_sum[i - 1] + lst[i])
    return cumulative_sum


def cumprod(lst: arr) -> list:
    """
    introduction
    ==========
    Sequence multiplication

    example
    ==========
    >>> cumprod([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    [1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
    >>>
    :param lst: list | tuple
    :return:
    """
    if not lst:
        return []
    cumulative_prod = [lst[0]]
    for i in range(1, len(lst)):
        cumulative_prod.append(cumulative_prod[i - 1] * lst[i])
    return cumulative_prod


def iroot(y: int, n: int) -> int:
    """
    introduction
    ==========
    Round after rooting \n
    [ʸ√x̄]

    example
    ==========
    >>> iroot(-2 ** 1000, 9)
    -2803995394640233757514136098827399
    >>>
    :param y: integer
    :param n: integer
    :return:
    """
    from math import log as _log
    i = False
    if y < 0 and n & 1:
        y = abs(y)
        i = True
    if y in [0, 1] or n == 1:
        return y
    elif n >= y.bit_length():
        return 1
    try:
        g = int(y ** (1 / n) + 0.5)
    except OverflowError:
        e = _log(y, 2) / n
        if e > 53:
            s = int(e - 53)
            g = int(2 ** (e - s) + 1) << s
        else:
            g = int(2 ** e)
    if g > 2 ** 50:
        p, x = -1, g
        while abs(x - p) >= 2:
            p, x = x, ((n - 1) * x + y // x ** (n - 1)) // n
    else:
        x = g
    t = x ** n
    while t < y:
        x += 1
        t = x ** n
    if t > y:
        x -= 1
    return -x if i else x


A = arrangement
C = combination
if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
