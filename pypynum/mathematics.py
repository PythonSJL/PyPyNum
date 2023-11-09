num = int | float | complex
real = int | float
arr = list | tuple


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
    5.333332933332714
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
        _sum += h * f(x) + (h / 2) * (f(x + h) + f(x))
    return _sum


def gamma(alpha: num) -> float:
    """
    Gamma function calculation
    >>> gamma(1.5)
    1.7724534351373467
    >>>
    :param alpha: float
    :return: float
    """
    from math import exp
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
