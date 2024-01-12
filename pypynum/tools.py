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
    if not isinstance(key, arr):
        raise TypeError("The key can only be a list")
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
                if retain is True:
                    result.append(indexes[index])
                pointer += len(indexes[index])
            else:
                break
        result.append(iterable[pointer:])
    elif isinstance(iterable, arr):
        for item in range(len(iterable)):
            if iterable[item] in key:
                result.append(iterable[pointer:item])
                if isinstance(iterable, arr):
                    pointer = item + 1
                if retain is True:
                    result.append(iterable[item])
        result.append(iterable[pointer:])
    else:
        raise TypeError("Iterable can only be a list, tuple or str")
    return result


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
