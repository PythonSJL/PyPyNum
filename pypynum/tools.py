arr = list | tuple
ite = list | tuple | str
real = int | float


def frange(start: real, stop: real, step: float = 1.0) -> list:
    """
    Float range
    >>> frange(0, 10, 1.5)
    [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0]
    >>>
    :param start: integer | float
    :param stop: integer | float
    :param step: float
    :return:
    """
    step = float(step)
    return [_ * step for _ in range(int(start / step), int(stop / step) + 1)]


def linspace(start: real, stop: real, number: int) -> list:
    """
    Linear space
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


def deduplicate(iterable: ite) -> ite:
    """
    Iterative object deduplication
    >>> deduplicate(["T", "h", "i", "s", " ", "i", "s", " ", "a", " ", "l", "i", "s", "t"])
    ['T', 'h', 'i', 's', ' ', 'a', 'l', 't']
    >>>
    :param iterable: list | tuple | string
    :return:
    """
    result = []
    for i in iterable:
        if i not in result:
            result.append(i)
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
    Data classification
    >>> classify([1, 2.3, 4 + 5j, 6 - 7j, 8.9, 0])
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


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
