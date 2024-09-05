import math
from .types import num, real


def bessel_j0(x: num) -> num:
    """
    Introduction
    ==========
    Calculate the Bessel function of the first kind of order 0, J_0(x).

    Example
    ==========
    >>> bessel_j0(1)
    0.7651976865579666
    >>>
    :param x: The argument of the Bessel function.
    :return: The Bessel function of the first kind of order 0 at x.
    """
    if x == 0:
        return 1.0
    result = 1.0
    term = 1.0
    k = 1
    x_squared = x * x
    while True:
        term *= -x_squared / (4 * k * k)
        temp = result
        result += term
        if result == temp:
            break
        k += 1
    return result


def bessel_j1(x: num) -> num:
    """
    Introduction
    ==========
    Calculate the Bessel function of the first kind of order 1, J_1(x).

    Example
    ==========
    >>> bessel_j1(1)
    0.44005058574493355
    >>>
    :param x: The argument of the Bessel function.
    :return: The Bessel function of the first kind of order 1 at x.
    """
    if x == 0:
        return 0.0
    result = 0.0
    term = 0.5
    k = 1
    x_squared = x * x
    while True:
        temp = result
        result += term
        if result == temp:
            break
        term *= -x_squared / (4 * k * (k + 1))
        k += 1
    return result * x


def bessel_jv(v: real, x: num) -> num:
    """
    Introduction
    ==========
    Calculate the Bessel function of the first kind of order v, J_v(x).

    Example
    ==========
    >>> bessel_jv(1, 1)
    0.44005058574493355
    >>>
    :param v: The order of the Bessel function.
    :param x: The argument of the Bessel function.
    :return: The Bessel function of the first kind of order v at x.
    """
    result = 0.0
    k = 0
    while True:
        try:
            term = (-1) ** k * (x / 2) ** (2 * k + v) / math.factorial(k) / math.gamma(k + v + 1)
            result += term
        except OverflowError:
            break
        except ValueError:
            pass
        k += 1
    return result


def bessel_i0(x: num) -> num:
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
    result = 1.0
    term = 1.0
    k = 0
    while True:
        k += 1
        term *= (x / 2) ** 2 / (k * k)
        temp = result
        result += term
        if result == temp:
            break
    return result


def bessel_i1(x: num) -> num:
    """
    Introduction
    ==========
    Calculate the modified Bessel function of the first kind of order 1, I_1(x).

    Example
    ==========
    >>> bessel_i1(1)
    0.565159103992485
    >>>
    :param x: The argument of the modified Bessel function, must be non-negative.
    :return: The modified Bessel function of the first kind of order 1 at x.
    """
    result = x / 2
    term = x / 2
    k = 1
    while True:
        term *= (x / 2) ** 2 / (k * (k + 1))
        temp = result
        result += term
        if result == temp:
            break
        k += 1
    return result


def bessel_iv(v: real, x: num) -> num:
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
            result += (x / 2) ** (2 * k + v) / math.factorial(k) / math.gamma(k + v + 1)
        except OverflowError:
            break
        except ValueError:
            pass
        k += 1
    return result
