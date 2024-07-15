import math
from .types import Union, arr, num, real, ite

__and = {0, 1, 4, 9, 16, 17, 25, 33, 36, 41, 49, 57, 64, 65, 68, 73, 81, 89, 97, 100, 105, 113,
         121, 129, 132, 137, 144, 145, 153, 161, 164, 169, 177, 185, 193, 196, 201, 209, 217, 225, 228, 233, 241, 249}
__mod = {0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 145, 160, 169,
         180, 196, 225, 241, 244, 256, 265, 289, 304, 324, 340, 361, 369, 385, 400, 409,
         436, 441, 481, 484, 496, 505, 529, 544, 576, 580, 585, 601, 625, 640, 649, 676}


def lower_gamma(s, x):
    """
    Introduction
    ==========
    Calculate the lower incomplete gamma function, denoted as γ(s, x) or P(s, x).

    Example
    ==========
    >>> lower_gamma(1, 2)
    0.8646647167633872
    >>>
    :param s: The shape parameter of the gamma function.
    :param x: The upper limit of the integration.
    :return: The lower incomplete gamma function of parameter s at x.
    """
    if x == 0:
        return x
    result = 0
    i = 0
    term = 1
    while term:
        try:
            term *= x / (s + i)
            temp = result
            result += term
            if result == temp:
                break
            i += 1
        except OverflowError:
            break
    return result * x ** (s - 1) * 2.718281828459045 ** (-x)


def upper_gamma(s, x):
    """
    Introduction
    ==========
    Calculate the upper incomplete gamma function, denoted as Q(s, x).

    Example
    ==========
    >>> upper_gamma(1, 2)
    0.1353352832366128
    >>>
    :param s: The shape parameter of the gamma function.
    :param x: The lower limit of the integration.
    :return: The upper incomplete gamma function of parameter s at x.
    """
    return math.gamma(s) - lower_gamma(s, x)


def bessel_iv(v, x):
    """
    Introduction
    ==========
    Calculate the modified Bessel function of the first kind of order v, I_v(x).

    Example
    ==========
    >>> bessel_iv(1, 1)
    0.565159103992485
    >>>
    :param v: The order of the modified Bessel function.
    :param x: The argument of the modified Bessel function.
    :return: The modified Bessel function of the first kind of order v at x.
    """
    result = 0
    k = 0
    while True:
        try:
            result += (x / 2) ** (2 * k + v) / factorial(k) / gamma(k + v + 1)
        except OverflowError:
            break
        except ValueError:
            pass
        k += 1
    return result


def bessel_i0(x):
    """
    Introduction
    ==========
    Calculate the modified Bessel function of the first kind of order 0, I_0(x).

    Example
    ==========
    >>> bessel_i0(1)
    1.2660658777520082
    >>>
    :param x: The argument of the modified Bessel function, must be non-negative.
    :return: The modified Bessel function of the first kind of order 0 at x.
    """
    return bessel_iv(0, x)


def sumprod(*arrays: arr) -> num:
    """
    Introduction
    ==========
    The sum of the products of the corresponding elements in multiple arrays

    Example
    ==========
    >>> sumprod([1, 2, 3], [4, 5, 6], [7, 8, 9])
    270
    >>>
    :param arrays: list | tuple
    :return:
    """
    return math.fsum(map(product, zip(*arrays)))


def xlogy(x: num, y: num) -> num:
    """
    Introduction
    ==========
    Compute the product of x and the logarithm of y for real and complex numbers.

    Example
    ==========
    >>> xlogy(2, 3)
    2.1972245773362196
    >>>
    :param x: Any real or complex number.
    :param y: A positive real number or a complex number with a positive real part,
        as logarithm is only defined for positive real values or complex numbers with a positive real part.
    :return: The result of x * log(y).
    """
    if isinstance(y, (int, float)) and y <= 0:
        raise ValueError("xlogy is not defined for y <= 0 when y is a real number")
    elif isinstance(y, complex) and y.real <= 0:
        raise ValueError("xlogy is not defined for y with a non-positive real part when y is a complex number")
    elif x == 0:
        return 0
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return x * math.log(y)
    else:
        from cmath import log
        return x * log(y)


def roll(seq: ite, shift: int) -> ite:
    """
    Introduction
    ==========
    Roll all elements of a sequence

    Example
    ==========
    >>> roll([1, 2, 3, 4, 5, 6, 7, 8, 9], 5)
    [5, 6, 7, 8, 9, 1, 2, 3, 4]
    >>>
    :param seq: list | tuple | string
    :param shift: integer
    :return:
    """
    shift = shift % len(seq)
    return seq[-shift:] + seq[:-shift]


def root(x: num, y: num) -> num:
    """
    Introduction
    ==========
    Root \n
    ʸ√x̄

    Example
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
    Introduction
    ==========
    Exponential \n
    eˣ

    Example
    ==========
    >>> exp(0.5)
    1.6487212707001282
    >>>
    :param x: integer | float
    :return:
    """
    return math.exp(x)


def ln(x: real) -> real:
    """
    Introduction
    ==========
    Natural logarithm \n
    logₑx

    Example
    ==========
    >>> ln(0.5)
    -0.6931471805599453
    >>>
    :param x: integer | float
    :return:
    """
    return math.log(x)


def gcd(*args: int) -> int:
    """
    Introduction
    ==========
    The greatest common factor of multiple integers

    Example
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
    Introduction
    ==========
    The least common multiple of multiple integers

    Example
    ==========
    >>> lcm(12, 18, 24)
    72
    >>>
    :param args: integer
    :return:
    """
    if not args:
        return 0
    try:
        f = math.lcm
    except AttributeError:
        def f(a, b):
            return a * b // math.gcd(a, b)
    lcm_value = args[0]
    for n in args[1:]:
        lcm_value = f(lcm_value, n)
    return lcm_value


def sin(x: real) -> real:
    """
    Sine
    :param x: integer | float
    :return:
    """
    return math.sin(x)


def cos(x: real) -> real:
    """
    Cosine
    :param x: integer | float
    :return:
    """
    return math.cos(x)


def tan(x: real) -> real:
    """
    Tangent
    :param x: integer | float
    :return:
    """
    return math.tan(x)


def csc(x: real) -> real:
    """
    Cosecant
    :param x: integer | float
    :return:
    """
    return 1 / sin(x)


def sec(x: real) -> real:
    """
    Secant
    :param x: integer | float
    :return:
    """
    return 1 / cos(x)


def cot(x: real) -> real:
    """
    Cotangent
    :param x: integer | float
    :return:
    """
    return cos(x) / sin(x)


def asin(x: real) -> real:
    """
    Arcsine
    :param x: integer | float
    :return:
    """
    return math.asin(x)


def acos(x: real) -> real:
    """
    Arccosine
    :param x: integer | float
    :return:
    """
    return math.acos(x)


def atan(x: real) -> real:
    """
    Arctangent
    :param x: integer | float
    :return:
    """
    return math.atan(x)


def acsc(x: real) -> real:
    """
    Arccosecant
    :param x: integer | float
    :return:
    """
    return asin(1 / x)


def asec(x: real) -> real:
    """
    Arcsecant
    :param x: integer | float
    :return:
    """
    return acos(1 / x)


def acot(x: real) -> real:
    """
    Arccotangent
    :param x: integer | float
    :return:
    """
    return atan(1 / x)


def sinh(x: real) -> real:
    """
    Hyperbolic Sine
    :param x: integer | float
    :return:
    """
    return math.sinh(x)


def cosh(x: real) -> real:
    """
    Hyperbolic Cosine
    :param x: integer | float
    :return:
    """
    return math.cosh(x)


def tanh(x: real) -> real:
    """
    Hyperbolic Tangent
    :param x: integer | float
    :return:
    """
    return math.tanh(x)


def csch(x: real) -> real:
    """
    Hyperbolic Cosecant
    :param x: integer | float
    :return:
    """
    return 1 / sinh(x)


def sech(x: real) -> real:
    """
    Hyperbolic Secant
    :param x: integer | float
    :return:
    """
    return 1 / cosh(x)


def coth(x: real) -> real:
    """
    Hyperbolic Cotangent
    :param x: integer | float
    :return:
    """
    return cosh(x) / sinh(x)


def asinh(x: real) -> real:
    """
    Hyperbolic Arcsine
    :param x: integer | float
    :return:
    """
    return math.asinh(x)


def acosh(x: real) -> real:
    """
    Hyperbolic Arccosine
    :param x: integer | float
    :return:
    """
    return math.acosh(x)


def atanh(x: real) -> real:
    """
    Hyperbolic Arctangent
    :param x: integer | float
    :return:
    """
    return math.atanh(x)


def acsch(x: real) -> real:
    """
    Hyperbolic Arccosecant
    :param x: integer | float
    :return:
    """
    return asinh(1 / x)


def asech(x: real) -> real:
    """
    Hyperbolic Arcsecant
    :param x: integer | float
    :return:
    """
    return acosh(1 / x)


def acoth(x: real) -> real:
    """
    Hyperbolic Arccotangent
    :param x: integer | float
    :return:
    """
    return atanh(1 / x)


def ptp(numbers: arr) -> num:
    """
    Introduction
    ==========
    Range of numbers

    Example
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
    Introduction
    ==========
    Median of numbers

    Example
    ==========
    >>> median([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    5.5
    >>>
    :param numbers: list | tuple
    :return:
    """
    data = sorted(numbers)
    n = len(data)
    if n & 1:
        return data[n // 2]
    else:
        return (data[n // 2 - 1] + data[n // 2]) / 2


def freq(data: arr) -> dict:
    """
    Introduction
    ==========
    Frequency of data

    Example
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
    Introduction
    ==========
    Mode of data

    Example
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
    Introduction
    ==========
    Average of numbers

    Example
    ==========
    >>> mean([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    5.5
    >>>
    :param numbers: list | tuple
    :return:
    """
    return math.fsum(numbers) / len(numbers)


def geom_mean(numbers: arr) -> num:
    """
    Introduction
    ==========
    The geometric mean of numbers

    Example
    ==========
    >>> geom_mean([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    4.528728688116765
    >>>
    :param numbers: list | tuple
    :return:
    """
    return root(product(numbers), len(numbers))


def square_mean(numbers: arr) -> num:
    """
    Introduction
    ==========
    Square mean of numbers

    Example
    ==========
    >>> square_mean([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    6.2048368229954285
    >>>
    :param numbers: list | tuple
    :return:
    """
    return math.sqrt(math.fsum(map(lambda n: n * n, numbers)) / len(numbers))


def harm_mean(numbers: arr) -> num:
    """
    Introduction
    ==========
    The harmonic mean of numbers

    Example
    ==========
    >>> harm_mean([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    3.414171521474055
    >>>
    :param numbers: list | tuple
    :return:
    """
    return 0 if 0 in numbers else len(numbers) / math.fsum(map(lambda n: 1 / n, numbers))


def raw_moment(data: arr, order: int) -> float:
    """
    Introduction
    ==========
    Calculate the origin moment of the sample

    Example
    ==========
    >>> raw_moment([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5)
    22082.5
    >>>
    :param data: list | tuple
    :param order: integer
    :return:
    """
    return math.fsum(map(lambda item: item ** order, data)) / len(data)


def central_moment(data: arr, order: int) -> float:
    """
    Introduction
    ==========
    Calculate the center moment of the sample

    Example
    ==========
    >>> central_moment([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5)
    0.0
    >>>
    :param data: list | tuple
    :param order: integer
    :return:
    """
    avg = mean(data)
    return math.fsum(map(lambda item: (item - avg) ** order, data)) / len(data)


def var(numbers: arr, dof: int = 0) -> num:
    """
    Introduction
    ==========
    Variance of numbers

    Example
    ==========
    >>> var([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    8.25
    >>>
    :param numbers: list | tuple
    :param dof: integer
    :return:
    """
    if dof >= len(numbers):
        return float("inf")
    avg = mean(numbers)

    def inner(item):
        d = item - avg
        return d * d

    return math.fsum(map(inner, numbers)) / (len(numbers) - dof)


def skew(data: arr) -> float:
    """
    Introduction
    ==========
    Calculate the skewness of the sample

    Example
    ==========
    >>> skew([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    0.0
    >>>
    :param data: list | tuple
    :return:
    """
    return central_moment(data, 3) / std(data) ** 3


def kurt(data: arr, fisher: bool = True) -> float:
    """
    Introduction
    ==========
    Calculate the kurtosis of the sample

    Example
    ==========
    >>> kurt([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    -1.2242424242424244
    >>>
    :param data: list | tuple
    :param fisher: boolean
    :return:
    """
    result = central_moment(data, 4) / std(data) ** 4
    if fisher:
        result -= 3
    return result


def std(numbers: arr, dof: int = 0) -> num:
    """
    Introduction
    ==========
    Standard deviation of numbers

    Example
    ==========
    >>> std([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    2.8722813232690143
    >>>
    :param numbers: list | tuple
    :param dof: integer
    :return:
    """
    return math.sqrt(var(numbers, dof=dof))


def cov(x: arr, y: arr, dof: int = 0) -> num:
    """
    Introduction
    ==========
    Covariance of numbers

    Example
    ==========
    >>> cov([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    2.0
    >>>
    :param x: list | tuple
    :param y: list | tuple
    :param dof: integer
    :return:
    """
    if dof >= min(len(x), len(y)):
        return float("inf")
    mean_x = mean(x)
    mean_y = mean(y)
    return math.fsum([(x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y)]) / (len(x) - dof)


def corr_coeff(x: arr, y: arr) -> num:
    """
    Introduction
    ==========
    The correlation coefficient of numbers

    Example
    ==========
    >>> corr_coeff([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    0.9999999999999998
    >>>
    :param x: list | tuple
    :param y: list | tuple
    :return:
    """
    if len(set(y)) == 1:
        return 1.0
    n = len(x)
    mean_x = mean(x)
    mean_y = mean(y)
    covariance = math.fsum((_x - mean_x) * (_y - mean_y) for _x, _y in zip(x, y)) / n
    std_x = math.sqrt(math.fsum((_x - mean_x) ** 2 for _x in x) / n)
    std_y = math.sqrt(math.fsum((_y - mean_y) ** 2 for _y in y) / n)
    return covariance / (std_x * std_y)


def coeff_det(x: arr, y: arr) -> num:
    """
    Introduction
    ==========
    The coefficient of determination for numbers

    Example
    ==========
    >>> coeff_det([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    0.9999999999999996
    >>>
    :param x: list | tuple
    :param y: list | tuple
    :return:
    """
    return corr_coeff(x, y) ** 2


def product(numbers: arr) -> num:
    """
    Introduction
    ==========
    Product of multiple numbers

    Example
    ==========
    >>> product([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    3628800
    >>>
    :param numbers: list | tuple
    :return:
    """
    if len(numbers) == 0:
        return 1
    elif 0 in numbers:
        return 0
    elif len(numbers) == 1:
        return numbers[0]
    elif len(numbers) <= 4096:
        result = numbers[0]
        for i in numbers[1:]:
            result *= i
        return result

    def prod(n):
        if len(n) == 1:
            return n[0]
        mid = len(n) // 2
        left_half = n[:mid]
        right_half = n[mid:]
        left_product = prod(left_half)
        right_product = prod(right_half)
        return left_product * right_product

    return prod(numbers)


def sigma(i: int, n: int, f) -> num:
    """
    Introduction
    ==========
    n \n
    Σ  f(x) \n
    i

    Example
    ==========
    >>> sigma(1, 10, lambda x: x ** x)
    10405071317
    >>>
    :param i: integer
    :param n: integer
    :param f: function
    :return:
    """
    return sum(map(f, range(i, n + 1)))


def pi(i: int, n: int, f) -> num:
    """
    Introduction
    ==========
    n \n
    Π  f(x) \n
    i

    Example
    ==========
    >>> pi(1, 10, lambda x: x ** x)
    215779412229418562091680268288000000000000000
    >>>
    :param i: integer
    :param n: integer
    :param f: function
    :return:
    """
    return product(list(map(f, range(i, n + 1))))


def deriv(f, x: real, h: real = 1e-6, *args, **kwargs) -> float:
    """
    Introduction
    ==========
    Derivative calculation

    Example
    ==========
    >>> deriv(lambda a: a ** 2, 2)
    4.000000000115023
    >>>
    :param f: function
    :param x: integer | float
    :param h: integer | float
    :return:
    """
    h = h / 2
    if args or kwargs:
        return (f(x + h, *args, **kwargs) - f(x - h, *args, **kwargs)) / (2 * h)
    else:
        return (f(x + h) - f(x - h)) / (2 * h)


def integ(f, x_start: real, x_end: real, n: int = 1000000, *args, **kwargs) -> float:
    """
    Introduction
    ==========
    Definite integral calculation

    Example
    ==========
    >>> integ(lambda a: a ** 2, 0, 2)
    2.6666666666666665
    >>>
    :param f: function
    :param x_start: integer | float
    :param x_end: integer | float
    :param n: integer | float
    :return:
    """
    h = (x_end - x_start) / n
    if args or kwargs:
        s = f(x_start, *args, **kwargs) + f(x_end, *args, **kwargs)
        s += math.fsum(map(lambda i: 4 * f(x_start + i * h, *args, **kwargs), range(1, n, 2)))
        s += math.fsum(map(lambda i: 2 * f(x_start + i * h, *args, **kwargs), range(2, n - 1, 2)))
    else:
        s = f(x_start) + f(x_end)
        s += math.fsum(map(lambda i: 4 * f(x_start + i * h), range(1, n, 2)))
        s += math.fsum(map(lambda i: 2 * f(x_start + i * h), range(2, n - 1, 2)))
    return s * h / 3


def beta(p: real, q: real) -> real:
    """
    Introduction
    ==========
    Beta function calculation \n
    Β(p, q)

    Example
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
    Introduction
    ==========
    Gamma function calculation \n
    Γ(s)

    Example
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
    Introduction
    ==========
    Integer factorial

    Example
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
    Introduction
    ==========
    A(n, r)

    Example
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
    Introduction
    ==========
    C(n, r)

    Example
    ==========
    >>> combination(10, 5)
    252
    >>>
    :param n: integer
    :param r: integer
    :return:
    """
    if r > n >> 1:
        return combination(n, n - r)
    return product(list(range(n, n - r, -1))) // product(list(range(r, 0, -1)))


def zeta(alpha: num) -> num:
    """
    Introduction
    ==========
    Zeta function calculation \n
    ζ(s)

    Example
    ==========
    >>> zeta(3)
    1.2020569031590942
    >>>
    :param alpha: integer | float | complex
    :return:
    """
    seq = map(lambda _: _ ** (-alpha), range(1, 1000000))
    if alpha.imag:
        return sum(seq)
    else:
        return math.fsum(seq)


def erf(x: real) -> float:
    """
    Introduction
    ==========
    Error function calculation

    Example
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
    Introduction
    ==========
    Sigmoid function calculation

    Example
    ==========
    >>> sigmoid(1)
    0.7310585786300049
    >>>
    :param x: integer | float
    :return:
    """
    return 1 / (1 + math.exp(-x))


def sign(x: num) -> num:
    """
    Introduction
    ==========
    Sign function calculation

    Example
    ==========
    >>> sign(10)
    1.0
    >>>
    :param x: integer | float | complex
    :return:
    """
    return 0.0 if x == 0 else x / abs(x)


def parity(x: int) -> int:
    """
    Introduction
    ==========
    Calculate (-1) ** n using bits

    Example
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
    Introduction
    ==========
    Sequence accumulation

    Example
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
    length = len(lst)
    for i in range(1, length):
        cumulative_sum.append(cumulative_sum[i - 1] + lst[i])
    return cumulative_sum


def cumprod(lst: arr) -> list:
    """
    Introduction
    ==========
    Sequence multiplication

    Example
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
    length = len(lst)
    for i in range(1, length):
        cumulative_prod.append(cumulative_prod[i - 1] * lst[i])
    return cumulative_prod


def iroot(y: int, n: int) -> int:
    """
    Introduction
    ==========
    Round after rooting \n
    [ʸ√x̄]

    Example
    ==========
    >>> iroot(-2 ** 1000, 9)
    -2803995394640233757514136098827399
    >>>
    :param y: integer
    :param n: integer
    :return:
    """
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
        e = math.log(y, 2) / n
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


def totient(n: int) -> int:
    """
    Introduction
    ==========
    Euler's totient function

    Example
    ==========
    >>> totient(100)
    40
    >>>
    :param n: integer
    :return:
    """
    if n == 1:
        return 1
    phi = n
    for p in range(2, int(math.sqrt(n)) + 1):
        if n % p == 0:
            phi = phi - phi // p
            while n % p == 0:
                n //= p
    if n > 1:
        phi = phi - phi // n
    return phi


def mod_order(a: int, n: int, b: int) -> int:
    """
    Introduction
    ==========
    The order of element b in the multiplication group of module a

    Example
    ==========
    >>> mod_order(101, 100, 99)
    100
    >>>
    :param a: integer
    :param n: integer
    :param b: integer
    :return:
    """
    p = 1
    c = b % a
    while p <= n and c != 1:
        c = c * b % a
        p += 1
    if p <= n:
        return p
    else:
        return -1


def primitive_root(a: int, single: bool = False) -> Union[int, list]:
    """
    Introduction
    ==========
    The order of element b in the multiplication group of module a

    Example
    ==========
    >>> primitive_root(11)
    [2, 6, 7, 8]
    >>>
    :param a: integer
    :param single: bool
    :return:
    """
    n = totient(a)
    p = []
    for b in range(2, a):
        if mod_order(a, n, b) == n:
            if single:
                return b
            p.append(b)
    return p


def normalize(data: arr, target: num = 1) -> arr:
    """
    Introduction
    ==========
    Scale all numbers proportionally until the total is exactly the target value

    Example
    ==========
    >>> normalize([1, 2, 3, 4, 5])
    [0.06666666666666667, 0.13333333333333333, 0.2, 0.26666666666666666, 0.3333333333333333]
    >>>
    :param data: list | tuple
    :param target: integer | float
    :return:
    """
    total = math.fsum(data)
    if total == 0 or target == 0:
        return type(data)([0]) * len(data)
    ratio = target / total
    return type(data)(map(lambda x: x * ratio, data))


def average(data, weights, expected=False):
    """
    Introduction
    ==========
    Calculate the weighted average or expected value of numbers

    Example
    ==========
    >>> average([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    3.25
    >>>
    :param data: list | tuple
    :param weights: list | tuple
    :param expected: bool
    :return:
    """
    if expected:
        weights = normalize(weights)
    result = math.fsum(map(lambda value: value[0] * value[1], zip(data, weights)))
    return result if expected else result / math.fsum(weights)


def exgcd(a: int, b: int) -> tuple:
    """
    Introduction
    ==========
    Extended Euclidean algorithm for calculating the maximum common factor and Bezier number

    Example
    ==========
    >>> exgcd(142, 857)
    (1, -344, 57)
    >>>
    :param a: integer
    :param b: integer
    :return:
    """
    if b == 0:
        return a, 1, 0
    else:
        x, y, g = 0, 1, b
        x1, y1, gcd_prev = 1, 0, a
        while g != 0:
            q = gcd_prev // g
            gcd_next, x_next, y_next = gcd_prev % g, x1 - q * x, y1 - q * y
            gcd_prev, x1, y1 = g, x, y
            g, x, y = gcd_next, x_next, y_next
        return gcd_prev, x1, y1


def crt(n: arr, a: arr) -> int:
    """
    Introduction
    ==========
    Using Chinese residual theory to solve congruence equations

    Example
    ==========
    >>> crt([9, 8, 7], [6, 5, 4])
    501
    >>>
    :param n: list | tuple
    :param a: list | tuple
    :return:
    """
    s = 0
    prod = product(n)
    for ni, ai in zip(n, a):
        if ni <= ai:
            return 0
        p = prod // ni
        g, x, y = exgcd(p, ni)
        if g != 1:
            return 0
        else:
            inv = x % ni
        s += ai * inv * p
    return s % prod


def isqrt(x: int) -> int:
    """
    Introduction
    ==========
    Round after square root \n
    [√x̄]

    Example
    ==========
    >>> isqrt(620448401733239439360000)
    787685471322
    >>>
    :param x: integer
    :return:
    """
    return iroot(x, 2)


def is_possibly_square(n: int) -> bool:
    """
    Introduction
    ==========
    Check if it is a possible square number

    Example
    ==========
    >>> is_possibly_square(123456 ** 789)
    True
    >>>
    :param n: integer
    :return:
    """
    if n < 0:
        return False
    if n % 720 not in __mod:
        return False
    if n & 0xFF not in __and:
        return False
    return True


def is_square(n: int) -> bool:
    """
    Introduction
    ==========
    Check if it is a square number

    Example
    ==========
    >>> is_square(123456 ** 789)
    False
    >>>
    :param n: integer
    :return:
    """
    if is_possibly_square(n):
        try:
            from math import isqrt as f
        except ImportError:
            f = isqrt
        sqrt = f(n)
        return sqrt * sqrt == n
    return False


A = arrangement
C = combination
if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
