from .types import arr, ite, real


def frange(start: real, stop: real, step: float = 1.0) -> list:
    """
    introduction
    ==========
    Float range

    example
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
    introduction
    ==========
    Linear space

    example
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
    introduction
    ==========
    Geometric space

    example
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


def deduplicate(iterable: ite) -> ite:
    """
    introduction
    ==========
    Data deduplication

    example
    ==========
    >>> deduplicate(["T", "h", "i", "s", " ", "i", "s", " ", "a", " ", "l", "i", "s", "t"])
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
    introduction
    ==========
    Data classification

    example
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
    introduction
    ==========
    Data splitting

    example
    ==========
    >>> split((1, 2, 2, 3, 3, 2, 2, 1, 2, 3, 4, 5, 6, 7, 8), [1, 3, 5], retain=True)
    [(), 1, (2, 2), 3, (), 3, (2, 2), 1, (2,), 3, (4,), 5, (6, 7, 8)]
    >>>
    :param iterable: list | tuple
    :param key: list | tuple
    :param retain: bool
    :return:
    """
    key = deduplicate(key)
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
                    result.append([iterable[item]])
        result.append(iterable[pointer:])
    else:
        raise TypeError("Iterable can only be a list, tuple or str")
    return result


def interpolation(data: arr, length: int) -> list:
    """
    introduction
    ==========
    One-dimensional data interpolation

    example
    ==========
    >>> interpolation((2, 4, 4, 2), 6)
    [2, 3.320000000000001, 4.160000000000004, 4.160000000000012, 3.3200000000000074, 2]
    >>>
    :param data:
    :param length:
    :return:
    """
    from .regression import linear_regression, parabolic_regression
    expr = [lambda x: sum([k * x ** (1 - n) for n, k in enumerate(linear_regression([0, 1], [data[0], data[1]]))])]
    for i in range(len(data) - 2):
        tmp = parabolic_regression(list(range(i, i + 3)), data[i:i + 3])
        expr.append(lambda x, coefficients=tmp: sum([k * x ** (2 - n) for n, k in enumerate(coefficients)]))
    expr.append(lambda x: sum([k * x ** (1 - n) for n, k in
                               enumerate(linear_regression([len(data) - 2, len(data) - 1], [data[-2], data[-1]]))]))
    result = linspace(0, len(data) - 1, length)
    for item in range(length):
        if int(result[item]) != result[item]:
            result[item] = (expr[int(result[item])](result[item]) + expr[int(result[item] + 1)](result[item])) / 2
        else:
            result[item] = data[int(result[item])]
    return result


def primality(n: int, iter_num: int = 10) -> bool:
    """
    introduction
    ==========
    Using the Miller Rabin method to test the primality of positive integers

    example
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
        m = m >> 1
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


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
