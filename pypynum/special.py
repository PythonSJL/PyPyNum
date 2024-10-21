import math
from .types import arr, num, real


def besselj0(x: num) -> num:
    """
    Introduction
    ==========
    Calculate the Bessel function of the first kind of order 0, J_0(x).

    Example
    ==========
    >>> besselj0(1)
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


def besselj1(x: num) -> num:
    """
    Introduction
    ==========
    Calculate the Bessel function of the first kind of order 1, J_1(x).

    Example
    ==========
    >>> besselj1(1)
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


def besseljv(v: real, x: num) -> num:
    """
    Introduction
    ==========
    Calculate the Bessel function of the first kind of order v, J_v(x).

    Example
    ==========
    >>> besseljv(1, 1)
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


def besseli0(x: num) -> num:
    """
    Introduction
    ==========
    Calculate the modified Bessel function of the first kind of order 0, I_0(x).

    Example
    ==========
    >>> besseli0(1)
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


def besseli1(x: num) -> num:
    """
    Introduction
    ==========
    Calculate the modified Bessel function of the first kind of order 1, I_1(x).

    Example
    ==========
    >>> besseli1(1)
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


def besseliv(v: real, x: num) -> num:
    """
    Introduction
    ==========
    Calculate the modified Bessel function of the first kind of order v, I_v(x).

    Example
    ==========
    >>> besseliv(1, 1)
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


def qpochhammer(a: num, q: num, n: num = None) -> num:
    """
    Introduction
    ==========
    Calculate the q-Pochhammer symbol (or q-shifted factorial) for given parameters.

    The q-Pochhammer symbol is defined as: (a; q)_n = (1 - a) * (1 - a*q) * (1 - a*q^2) * ... * (1 - a*q^(n-1))

    Example
    ==========
    >>> qpochhammer(2 + 1j, 0.5 + 0.1j, 2 + 1j)
    (-0.33353429405776575+1.8573191887407854j)
    >>>
    :param a: The base parameter of the q-Pochhammer symbol.
    :param q: The q-parameter, must have an absolute value less than 1 for convergence.
    :param n: The number of terms in the product. If None, it defaults to infinity. If n is a complex number with a
        non-zero imaginary part or a non-integer real part, the function returns the ratio of two q-Pochhammer symbols.
    :return: The value of the q-Pochhammer symbol.
    """
    if n is None:
        n = float("inf")
    elif isinstance(n, (float, complex)):
        if n.imag or not n.real.is_integer():
            return qpochhammer(a, q) / qpochhammer(a * q ** n, q)
        else:
            n = int(n.real)
    if n == float("inf") and abs(q) >= 1:
        raise ValueError("The absolute value of q must be less than 1 to ensure convergence for infinite terms.")
    result = 1
    k = 0
    while k < n:
        term = 1 - a * (q ** k)
        if term == 1:
            break
        previous = result
        result *= term
        if result == previous:
            break
        k += 1
    return result


def qfactorial(n: num, q: num) -> num:
    """
    Introduction
    ==========
    Calculate the q-factorial of a given number n using the q-Pochhammer symbol.

    The q-factorial is defined as: [n]_q! = (q; q)_n / (1 - q)^n

    Example
    ==========
    >>> qfactorial(2 + 1j, 0.5 + 0.1j)
    (1.0769270963525002+0.6870695642029011j)
    >>>
    :param n: The number for which to calculate the q-factorial.
    :param q: The q-parameter, must have an absolute value less than 1 for convergence.
    :return: The value of the q-factorial.
    """
    if abs(q) >= 1:
        raise ValueError("The absolute value of q must be less than 1 to ensure convergence.")
    qpoch_q = qpochhammer(q, q)
    qpoch_q_pow_n = qpochhammer(q ** (1 + n), q)
    result = qpoch_q / ((1 - q) ** n * qpoch_q_pow_n)
    return result


def qgamma(n: num, q: num) -> num:
    """
    Introduction
    ==========
    Calculate the q-gamma function for a given number n using the q-Pochhammer symbol.

    The q-gamma function is defined as: Γ_q(n) = (q; q)_∞ / (q^n; q)_∞ * (1 - q)^(1 - n)

    Example
    ==========
    >>> qgamma(2 + 1j, 0.5 + 0.1j)
    (0.7815732121286768+0.23898083786505578j)
    >>>
    :param n: The number for which to calculate the q-gamma function.
    :param q: The q-parameter, must have an absolute value less than 1 for convergence.
    :return: The value of the q-gamma function.
    """
    if abs(q) >= 1:
        raise ValueError("The absolute value of q must be less than 1 to ensure convergence.")
    if n == 0:
        return 1 / (1 - q)
    qpoch_q = qpochhammer(q, q)
    qpoch_q_pow_n = qpochhammer(q ** n, q)
    result = ((1 - q) ** (1 - n) * qpoch_q) / qpoch_q_pow_n
    return result


def qbinomial(n: num, m: num, q: num) -> num:
    """
    Introduction
    ==========
    Calculate the q-binomial coefficient for given parameters using the specified formula.

    The q-binomial coefficient is defined as:
    (n choose m)_q = ((q^(m+1); q)_∞ * (q^(-m+n+1); q)_∞) / ((q; q)_∞ * (q^(n+1); q)_∞)

    Example
    ==========
    >>> qbinomial(3, 2, 0.5)
    1.75
    >>>
    :param n: The upper index of the q-binomial coefficient.
    :param m: The lower index of the q-binomial coefficient.
    :param q: The q-parameter, must have an absolute value less than 1 for convergence.
    :return: The value of the q-binomial coefficient.
    """
    if abs(q) >= 1:
        raise ValueError("The absolute value of q must be less than 1 to ensure convergence.")
    divisor = qpochhammer(q, q) * qpochhammer(q ** (n + 1), q)
    if not divisor:
        return float("inf")
    return qpochhammer(q ** (m + 1), q) * qpochhammer(q ** (-m + n + 1), q) / divisor