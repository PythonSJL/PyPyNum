import math

arr = list | tuple
num = int | float | complex
real = int | float


def exp(x: real) -> real:
    """
    Exponential
    :param x: int | float
    :return:
    """
    return math.exp(x)


def ln(x: real) -> real:
    """
    Natural logarithm
    :param x: int | float
    :return:
    """
    return math.log(x)


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
    Range of numbers
    >>> ptp([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    9
    >>>
    :param numbers: list | tuple
    :return:
    """
    return max(numbers) - min(numbers)


def median(numbers: arr) -> num:
    """
    Median of numbers
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
    Frequency of data
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


def mode(data: arr) -> any:
    """
    Mode of data
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
    Average of numbers
    >>> mean([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    5.5
    >>>
    :param numbers: list | tuple
    :return:
    """
    return sum(numbers) / len(numbers)


def var(numbers: arr) -> num:
    """
    Variance of numbers
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
    Standard deviation of numbers
    >>> std([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    2.8722813232690143
    >>>
    :param numbers: list | tuple
    :return:
    """
    return var(numbers) ** 0.5


def product(numbers: arr) -> num:
    """
    Product of multiple numbers
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
    Σ
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
    Π
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
    Derivative calculation
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
    Definite integral calculation
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


def gamma(alpha: real) -> float:
    """
    Gamma function calculation
    >>> gamma(1.5)
    0.8862267175684345
    >>>
    :param alpha: integer | float
    :return:
    """
    return definite_integral(lambda x: x ** (alpha - 1) * exp(-x), 0, 1000)


def factorial(n: int) -> int:
    """
    Integer factorial
    >>> factorial(10)
    3628800
    >>>
    :param n: integer
    :return:
    """
    return product(list(range(1, n + 1)))


def arrangement(n: int, r: int) -> int:
    """
    A(n, r)
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
    C(n, r)
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
    Zeta function calculation
    >>> zeta(2)
    1.6449339668472496
    >>>
    :param alpha: integer | float
    :return:
    """
    return sum([1 / _ ** alpha for _ in range(1, 10000000)])


def gaussian(x: real, _mu: real = 0, _sigma: real = 1) -> float:
    """
    Gaussian function calculation
    >>> gaussian(0)
    0.3989422804014327
    >>>
    :param x: integer | float
    :param _mu: integer | float
    :param _sigma: integer | float
    :return:
    """
    return exp(-(x - _mu) ** 2 / (2 * _sigma ** 2)) / (_sigma * 2.5066282746310002)


def sigmoid(x: real):
    """
    Sigmoid function calculation
    >>> sigmoid(1)
    0.7310585786300049
    >>>
    :param x: int | float
    :return:
    """
    return 1 / (1 + exp(-x))


def __parse(expression: str) -> list:
    """
    None
    :param expression: string
    :return:
    """
    length = len(expression)
    head = 0
    pointer = 0
    result = []
    while True:
        if pointer >= length - 1:
            if head <= length - 1:
                result.append(expression[head:])
            return result
        elif expression[pointer] == "(":
            tail = 0
            depth = 1
            for p in range(pointer + 1, length):
                if expression[p] == "(":
                    depth += 1
                elif expression[p] == ")":
                    depth -= 1
                if depth == 0:
                    tail = p
                    break
            result.append(__parse(expression[pointer + 1:tail]))
            head = tail + 1
            pointer = tail
        elif expression[pointer] in "+-":
            if head != pointer:
                result.append(expression[head:pointer])
                result.append(expression[pointer])
                head = pointer + 1
        elif expression[pointer] in "*/":
            if head != pointer:
                result.append(expression[head:pointer])
            if expression[pointer + 1] == expression[pointer]:
                result.append(expression[pointer:pointer + 2])
                pointer += 1
                head = pointer + 1
            else:
                result.append(expression[pointer:pointer + 1])
                head = pointer + 1
        pointer += 1


def parse(mathematical_expression: str) -> list:
    """
    Python Mathematical Expression Analysis
    >>> parse("(x + y - z) * (a - b + c)")
    [['x', '+', 'y', '-', 'z'], '*', ['a', '-', 'b', '+', 'c']]
    >>>
    :param mathematical_expression: string
    :return:
    """
    expression = mathematical_expression.replace(" ", "")
    depth = 0
    for p in range(len(expression)):
        if expression[p] == "(":
            depth += 1
        elif expression[p] == ")":
            depth -= 1
        if depth < 0:
            raise ValueError("The number of right parentheses is greater than the number of left parentheses")
    if depth > 0:
        raise ValueError("The number of left parentheses is greater than the number of right parentheses")
    return __parse(expression)


A = arrangement
C = combination
if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
