from .types import Union, arr, ite, real


def frange(start: real, stop: real, step: float = 1.0) -> list:
    """
    Introduction
    ==========
    Float range

    Example
    ==========
    >>> frange(0, 10, 1.5)
    [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0]
    >>>
    :param start: integer | float
    :param stop: integer | float
    :param step: float
    :return:
    """
    if isinstance(step, int):
        step = float(step)
    result = [_ * step for _ in range(int(start / step), int(stop / step + 1))]
    return result


def linspace(start: real, stop: real, number: int) -> list:
    """
    Introduction
    ==========
    Linear space

    Example
    ==========
    >>> linspace(2, 3, 4)
    [2.0, 2.3333333333333335, 2.6666666666666665, 3.0]
    >>>
    :param start: integer | float
    :param stop: integer | float
    :param number: integer
    :return:
    """
    if number == 1:
        return [start]
    elif number == 2:
        return [start, stop]
    else:
        step = (stop - start) / (number - 1)
        return [start + i * step for i in range(number)]


def geomspace(start: real, stop: real, number: int) -> list:
    """
    Introduction
    ==========
    Geometric space

    Example
    ==========
    >>> geomspace(2, 3, 4)
    [2.0, 2.2894284851066637, 2.620741394208897, 3.0]
    >>>
    :param start: integer | float
    :param stop: integer | float
    :param number: integer
    :return:
    """
    if number == 1:
        return [start]
    elif number == 2:
        return [start, stop]
    else:
        step = (stop / start) ** (1 / (number - 1))
        return [start * step ** i for i in range(number)]


def dedup(iterable: ite) -> ite:
    """
    Introduction
    ==========
    Data deduplication

    Example
    ==========
    >>> dedup(["T", "h", "i", "s", " ", "i", "s", " ", "a", " ", "l", "i", "s", "t"])
    ['T', 'h', 'i', 's', ' ', 'a', 'l', 't']
    >>>
    :param iterable: list | tuple | string
    :return:
    """
    result = []
    for item in iterable:
        if item not in result:
            result.append(item)
    if isinstance(iterable, list):
        return result
    elif isinstance(iterable, tuple):
        return tuple(result)
    elif isinstance(iterable, str):
        return "".join(result)
    else:
        raise TypeError("Iterable can only be a list, tuple or str")


def classify(array: arr) -> dict:
    """
    Introduction
    ==========
    Data classification

    Example
    ==========
    >>> classify((1, 2.3, 4 + 5j, 6 - 7j, 8.9, 0))
    {<class 'int'>: [1, 0], <class 'float'>: [2.3, 8.9], <class 'complex'>: [(4+5j), (6-7j)]}
    >>>
    :param array: list | tuple
    :return:
    """
    result = {}
    for item in array:
        _type = type(item)
        if _type in result:
            result[_type].append(item)
        else:
            result[_type] = [item]
    return result


def split(iterable: ite, key: arr, retain: bool = False) -> list:
    """
    Introduction
    ==========
    Data splitting

    Example
    ==========
    >>> split((1, 2, 2, 3, 3, 2, 2, 1, 2, 3, 4, 5, 6, 7, 8), [1, 3, 5], retain=True)
    [(), (1,), (2, 2), (3,), (), (3,), (2, 2), (1,), (2,), (3,), (4,), (5,), (6, 7, 8)]
    >>>
    :param iterable: list | tuple
    :param key: list | tuple
    :param retain: bool
    :return:
    """
    key = dedup(key)
    if not isinstance(retain, bool):
        raise TypeError("Only Boolean values can be used to determine whether to retain the key")
    if not isinstance(key, (list, tuple)):
        raise TypeError("The parameter 'key' can only be a list or tuple")
    result = []
    pointer = 0
    if isinstance(iterable, str):
        if key == [""]:
            raise ValueError("When iterable is a string, the key cannot be a list of empty strings")
        while True:
            indexes = {}
            for k in key:
                index = iterable.find(k, pointer)
                if index != -1 and index not in indexes:
                    indexes[index] = k
            if indexes:
                index = min(indexes)
                result.append(iterable[pointer:index])
                pointer = index
                if retain is True:
                    result.append(indexes[index])
                pointer += len(indexes[index])
            else:
                break
        result.append(iterable[pointer:])
    elif isinstance(iterable, (list, tuple)):
        for item in range(len(iterable)):
            if iterable[item] in key:
                result.append(iterable[pointer:item])
                if isinstance(iterable, (list, tuple)):
                    pointer = item + 1
                if retain is True:
                    result.append(iterable[item:item + 1])
        result.append(iterable[pointer:])
    else:
        raise TypeError("Iterable can only be a list, tuple or str")
    return result


def interp(data: arr, length: int) -> list:
    """
    Introduction
    ==========
    One-dimensional data interpolation

    Example
    ==========
    >>> interp((2, 4, 4, 2), 6)
    [2, 3.320000000000001, 4.160000000000004, 4.160000000000012, 3.3200000000000074, 2]
    >>>
    :param data:
    :param length:
    :return:
    """
    from .regression import lin_reg, par_reg
    expr = [lambda x: sum([k * x ** (1 - n) for n, k in enumerate(lin_reg([0, 1], [data[0], data[1]]))])]
    for i in range(len(data) - 2):
        tmp = par_reg(list(range(i, i + 3)), data[i:i + 3])
        expr.append(lambda x, coefficients=tmp: sum([k * x ** (2 - n) for n, k in enumerate(coefficients)]))
    expr.append(lambda x: sum([k * x ** (1 - n) for n, k in
                               enumerate(lin_reg([len(data) - 2, len(data) - 1], [data[-2], data[-1]]))]))
    result = linspace(0, len(data) - 1, length)
    for item in range(length):
        if int(result[item]) != result[item]:
            result[item] = (expr[int(result[item])](result[item]) + expr[int(result[item] + 1)](result[item])) / 2
        else:
            result[item] = data[int(result[item])]
    return result


def primality(n: int, iter_num: int = 10) -> bool:
    """
    Introduction
    ==========
    Using the Miller Rabin method to test the primality of positive integers

    Example
    ==========
    >>> primality(2 ** 4423 - 1)
    True
    >>>
    :param n: integer
    :param iter_num: integer
    :return:
    """
    from random import randint
    if n == 2:
        return True
    elif n & 1 == 0 or n < 2:
        return False
    m, s = n - 1, 0
    while m & 1 == 0:
        m >>= 1
        s += 1
    for _ in range(iter_num):
        b = pow(randint(2, n - 1), m, n)
        if b == 1 or b == n - 1:
            continue
        for __ in range(s - 1):
            b = b * b % n
            if b == n - 1:
                break
        else:
            return False
    return True


def generate_primes(limit: int) -> list:
    """
    Introduction
    ==========
    Generate all prime numbers within the specified limit using linear filtering method

    Example
    ==========
    >>> generate_primes(100)
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    >>>
    :param limit: integer
    :return:
    """
    primes = list(range(limit + 1))
    primes[1] = 0
    p = 2
    s = 4
    while s <= limit:
        if primes[p]:
            for i in range(s, limit + 1, p):
                primes[i] = 0
        p += 1
        s = p * p
    return [x for x in primes if x != 0]


def generate_semiprimes(limit: int) -> list:
    """
    Introduction
    ==========
    Generate all semiprime numbers within the specified limit

    Example
    ==========
    >>> generate_semiprimes(64)
    [4, 6, 9, 10, 14, 15, 21, 22, 25, 26, 33, 34, 35, 38, 39, 46, 49, 51, 55, 57, 58, 62]
    >>>
    :param limit: integer
    :return:
    """
    primes = generate_primes(limit // 2)
    length = len(primes)
    semiprimes = []
    for i in range(length):
        first = primes[i]
        maximum = limit // first
        for j in range(i, length):
            second = primes[j]
            if second > maximum:
                break
            semiprimes.append(first * second)
    return sorted(semiprimes)


def prime_factors(integer: int, dictionary: bool = False, pollard_rho: bool = True) -> Union[list, dict]:
    """
    Introduction
    ==========
    Using the Pollard Rho method to decompose prime factors of positive integers

    Example
    ==========
    >>> prime_factors(2305567963945518424753102147331756070)
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    >>>
    :param integer: integer
    :param dictionary: bool
    :param pollard_rho: bool
    :return:
    """
    from math import gcd
    from random import randint

    def normal(n):
        factors = []
        divisor = 2
        while n > 1:
            tmp = divmod(n, divisor)
            if tmp[1]:
                divisor += 1
            else:
                factors.append(divisor)
                n = tmp[0]
        return factors

    def pollard(x, c):
        i, k = 1, 2
        x0 = randint(0, x)
        y = x0
        while 1:
            i += 1
            x0 = (x0 * x0 % x + c) % x
            d = gcd(y - x0, x)
            if d != 1 and d != x:
                return d
            if y == x0:
                return x
            if i == k:
                y = x0
                k += k

    def recursion(n):
        result = []
        if primality(n):
            return [n]
        p = n
        while p >= n:
            p = pollard(p, randint(1, n - 1))
        result.extend(recursion(p))
        result.extend(recursion(n // p))
        return result

    if integer > 0:
        output = recursion(integer) if pollard_rho else normal(integer)
        if dictionary:
            return {number: output.count(number) for number in set(output)}
        else:
            return sorted(output)
    else:
        return []


def magic_square(n):
    """
    Introduction
    ==========
    Generate a magic square of a specified order

    Example
    ==========
    >>> magic_square(4)
    [[16, 2, 3, 13], [5, 11, 10, 8], [9, 7, 6, 12], [4, 14, 15, 1]]
    >>>
    :param n: integer
    :return:
    """
    if not isinstance(n, int) or n < 3:
        raise ValueError("The order of the magic square must be an integer greater than or equal to three")
    result = [[0] * n for _ in range(n)]

    def odd():
        i, j = 0, n // 2
        for m in range(1, n * n + 1):
            result[i][j] = m
            new_i, new_j = (i - 1) % n, (j + 1) % n
            if result[new_i][new_j] != 0:
                new_i, new_j = (i + 1) % n, j
            i, j = new_i, new_j

    def single_even():
        half = n // 2
        left = magic_square(half)
        x = n ** 2 // 4
        k = (n - 2) // 4
        for i in range(half):
            for j in range(half):
                result[i][j] = left[i][j]
                result[i][j + half] = left[i][j] + 2 * x
                result[i + half][j] = left[i][j] + 3 * x
                result[i + half][j + half] = left[i][j] + x
        for i in range(half):
            for j in range(n):
                if j > (half + k + 1):
                    result[i][j], result[i + half][j] = result[i + half][j], result[i][j]
                if j < k:
                    result[i][j], result[i + half][j] = result[i + half][j], result[i][j]
        result[k][0], result[k + half][0] = result[k + half][0], result[k][0]
        result[k][k], result[k + half][k] = result[k + half][k], result[k][k]

    def double_even():
        c1 = 1
        c2 = n * n
        for i in range(n):
            for j in range(n):
                if i % 4 == j % 4 or (i + j) % 4 == 3:
                    result[i][j] = c2
                else:
                    result[i][j] = c1
                c2 -= 1
                c1 += 1

    if n & 1:
        odd()
    elif n & 3:
        single_even()
    else:
        double_even()
    return result


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
