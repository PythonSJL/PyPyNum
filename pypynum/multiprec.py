from decimal import Decimal, getcontext as __
from fractions import Fraction

from .types import prec

__context = __()
del __


def frac2dec(frac: Fraction, sigfigs: int) -> Decimal:
    __context.prec = sigfigs
    return Decimal(frac.numerator) / Decimal(frac.denominator)


def mp_e(sigfigs: int, method: str = "series") -> Decimal:
    def series_e():
        nonlocal sigfigs
        sigfigs = sigfigs + 5
        __context.prec = sigfigs
        e = Decimal(0)
        one = Decimal(1)
        factorial = one
        n = 0
        current = one
        eps = Decimal(10) ** -sigfigs
        while current >= eps:
            e += current
            n += 1
            factorial *= n
            current = one / factorial
        __context.prec -= 5
        return +e

    def spigot_e():
        a = [1] * (sigfigs + 1)
        e = ["2."]
        for _ in range(sigfigs):
            carry = 0
            for i in range(sigfigs, -1, -1):
                carry, a[i] = divmod(a[i] * 10 + carry, i + 2)
            e.append(str(carry))
        return Decimal("".join(e))

    if method == "series":
        return series_e()
    elif method == "spigot":
        return spigot_e()
    else:
        raise ValueError("Invalid method. Use 'series' or 'spigot'")


def mp_pi(sigfigs: int, method: str = "chudnovsky") -> Decimal:
    def chudnovsky_pi():
        nonlocal sigfigs
        sigfigs = sigfigs + 5
        __context.prec = sigfigs
        c = 426880 * Decimal(10005).sqrt()
        m = 1
        _l = 13591409
        x = 1
        k = 6
        s = _l
        const = 14
        end = int(sigfigs / const) + 1
        for i in range(1, end):
            m = (k ** 3 - 16 * k) * m // (i ** 3)
            _l += 545140134
            x *= -262537412640768000
            s += Decimal(m * _l) / x
            k += 12
        pi = c / s
        __context.prec -= 5
        return +pi

    def bbp_pi():
        nonlocal sigfigs
        sigfigs = sigfigs + 5
        __context.prec = sigfigs
        d4 = Decimal(4)
        d2 = Decimal(2)
        d1 = Decimal(1)
        pi = Decimal(0)
        k = 0
        p16 = 1
        k8 = 0
        while True:
            pi += (d4 / (k8 + 1) - d2 / (k8 + 4) - d1 / (k8 + 5) - d1 / (k8 + 6)) / p16
            k += 1
            k8 += 8
            if k > sigfigs:
                break
            p16 <<= 4
        __context.prec -= 5
        return +pi

    if method == "chudnovsky":
        return chudnovsky_pi()
    elif method == "bbp":
        return bbp_pi()
    else:
        raise ValueError("Invalid method. Use 'chudnovsky' or 'bbp'")


def mp_phi(sigfigs: int, method: str = "algebraic") -> Decimal:
    def algebraic_phi():
        __context.prec = sigfigs
        one = Decimal(1)
        five = Decimal(5)
        two = Decimal(2)
        return (one + five.sqrt()) / two

    def newton_phi():
        nonlocal sigfigs
        sigfigs = sigfigs + 5
        __context.prec = sigfigs
        one = Decimal(1)
        two = Decimal(2)
        x = Decimal("1.5")
        eps = one / Decimal(10 ** sigfigs)
        while True:
            x_new = x - (x ** 2 - x - one) / (two * x - one)
            if abs(x_new - x) < eps:
                break
            x = x_new
        __context.prec -= 5
        return +x

    if method == "algebraic":
        return algebraic_phi()
    elif method == "newton":
        return newton_phi()
    else:
        raise ValueError("Invalid method. Use 'algebraic' or 'newton'")


def mp_sin(x: prec, sigfigs: int) -> Decimal:
    x = Decimal(x)
    sigfigs = sigfigs + 5
    __context.prec = sigfigs
    sin_x = Decimal(0)
    pi = mp_pi(sigfigs)
    x = x % (2 * pi)
    x_squared = x * x
    term = x
    n = 0
    eps = Decimal(10) ** -sigfigs
    while abs(term) >= eps:
        sin_x += term
        n += 1
        term = -term * x_squared / ((2 * n) * (2 * n + 1))
    __context.prec -= 5
    return +sin_x


def mp_cos(x: prec, sigfigs: int) -> Decimal:
    x = Decimal(x)
    sigfigs = sigfigs + 5
    __context.prec = sigfigs
    cos_x = Decimal(0)
    pi = mp_pi(sigfigs)
    x = x % (2 * pi)
    x_squared = x * x
    term = Decimal(1)
    n = 0
    eps = Decimal(10) ** -sigfigs
    while abs(term) >= eps:
        cos_x += term
        n += 1
        term = -term * x_squared / ((2 * n - 1) * (2 * n))
    __context.prec -= 5
    return +cos_x


def mp_ln(x: prec, sigfigs: int, builtin: bool = True) -> Decimal:
    if builtin:
        __context.prec = sigfigs
        return Decimal(x).ln()
    else:
        x = Decimal(x)
        if x <= 0:
            raise ValueError("Natural logarithm is not defined for x <= 0")
        sigfigs = sigfigs + 5
        __context.prec = sigfigs
        sign = -1
        if x > 1:
            x = 1 / x
            sign = 1
        ln_x = Decimal(0)
        term = Decimal(1)
        dx = 1 - x
        k = 1
        eps = Decimal(10) ** -sigfigs
        while abs(term) > eps:
            term *= dx
            ln_x += term / k
            k += 1
        __context.prec -= 5
        return ln_x * sign


def mp_log(x: prec, base: prec, sigfigs: int, builtin: bool = True) -> Decimal:
    sigfigs = sigfigs + 5
    if builtin:
        __context.prec = sigfigs
        x_dec = Decimal(x)
        base_dec = Decimal(base)
        log_x_base = x_dec.ln() / base_dec.ln()
    else:
        if x <= 0:
            raise ValueError("Logarithm is not defined for x <= 0")
        if base <= 0 or base == 1:
            raise ValueError("Logarithm base must be greater than 0 and not equal to 1")
        ln_x = mp_ln(x, sigfigs, False)
        ln_base = mp_ln(base, sigfigs, False)
        log_x_base = ln_x / ln_base
    __context.prec -= 5
    return +log_x_base
