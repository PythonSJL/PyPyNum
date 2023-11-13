from .Matrix import Matrix
from .Tensor import Tensor

ite = list | tuple | str


def copy(data):
    """
    Deep replication data
    >>> copy([1, "2", b"3", (4.4, 5 + 5j), [[6, 7], [8, 9]], True])
    [1, '2', b'3', (4.4, (5+5j)), [[6, 7], [8, 9]], True]
    >>>
    :param data: Any
    :return:
    """
    if isinstance(data, list):
        return [copy(i) for i in data]
    elif isinstance(data, tuple):
        return tuple(copy(i) for i in data)
    elif isinstance(data, (str, bytes, bytearray, int, float, complex, bool)):
        return data
    elif isinstance(data, Matrix):
        return Matrix(copy(data.data))
    elif isinstance(data, Tensor):
        return Tensor(copy(data.data))
    elif data is None:
        return data
    else:
        raise TypeError(
            "Data can only be Matrix, Tensor, list, tuple, str, bytes, bytearray, int, float, complex, bool or None"
        )


def deduplicate(iterable: ite) -> ite:
    """
    Iterative object deduplication
    >>> deduplicate(["T", "h", "i", "s", " ", "i", "s", " ", "a", " ", "l", "i", "s", "t"])
    ['T', 'h', 'i', 's', ' ', 'a', 'l', 't']
    >>>
    :param iterable: list | tuple | str
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


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
