import math
from .types import arr, num, real


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


def hyppfq(a: arr, b: arr, z: num) -> num:
    """
    Introduction
    ==========
    Calculate the generalized hypergeometric function pFq(a; b; z),
    which is a generalization of many named special functions.

    Example
    ==========
    >>> hyppfq([1, 2], [3], 0.5)
    1.545177444479562
    >>>
    :param a: List of numerator parameters of the hypergeometric function.
    :param b: List of denominator parameters of the hypergeometric function.
    :param z: The argument of the hypergeometric function.
    :return: The value of the hypergeometric function pFq(a; b; z).
    """
    p = len(a)
    q = len(b)
    result = 0
    numerator = [1] * p
    denominator = [1] * q
    z_power = 1
    factorial = 1
    n = 0
    while True:
        term = 1
        for i in range(p):
            term *= numerator[i]
        for j in range(q):
            term /= denominator[j]
        term *= z_power / factorial
        previous = result
        result += term
        if previous == result:
            break
        for i in range(p):
            numerator[i] *= (a[i] + n)
        for j in range(q):
            denominator[j] *= (b[j] + n)
        z_power *= z
        n += 1
        factorial *= n
    return result


def hyp0f1(b0: num, z: num) -> num:
    """
    Introduction
    ==========
    Calculate the hypergeometric function 0F1, which is a special case of the generalized hypergeometric function.

    Example
    ==========
    >>> hyp0f1(1, 0.5)
    1.5660829297563503
    >>>
    :param b0: The single parameter of the hypergeometric function.
    :param z: The argument of the hypergeometric function.
    :return: The value of the hypergeometric function 0F1(b0; z).
    """
    return hyppfq([], [b0], z)


def hyp1f1(a0: num, b0: num, z: num) -> num:
    """
    Introduction
    ==========
    Calculate the hypergeometric function 1F1, also known as the confluent hypergeometric function of the first kind.

    Example
    ==========
    >>> hyp1f1(1, 1, 1)
    2.7182818284590455
    >>>
    :param a0: The single numerator parameter of the hypergeometric function.
    :param b0: The single denominator parameter of the hypergeometric function.
    :param z: The argument of the hypergeometric function.
    :return: The value of the hypergeometric function 1F1(a0; b0; z).
    """
    return hyppfq([a0], [b0], z)


def hyp2f1(a0: num, a1: num, b0: num, z: num) -> num:
    """
    Introduction
    ==========
    Calculate the hypergeometric function 2F1, which is a common form of the generalized hypergeometric function.

    Example
    ==========
    >>> hyp2f1(1, 1, 1, 0.5)
    2.0
    >>>
    :param a0: The first numerator parameter of the hypergeometric function.
    :param a1: The second numerator parameter of the hypergeometric function.
    :param b0: The single denominator parameter of the hypergeometric function.
    :param z: The argument of the hypergeometric function.
    :return: The value of the hypergeometric function 2F1(a0, a1; b0; z).
    """
    return hyppfq([a0, a1], [b0], z)
