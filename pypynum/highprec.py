from decimal import Decimal, getcontext as __

__context = __()
del __


def calc_e(digits: int, method: str = "series") -> Decimal:
    def spigot_e():
        a = [1] * (digits + 1)
        e = ["2."]
        for _ in range(digits):
            carry = 0
            for i in range(digits, -1, -1):
                carry, a[i] = divmod(a[i] * 10 + carry, i + 2)
            e.append(str(carry))
        return Decimal("".join(e))

    def series_e():
        nonlocal digits
        digits = digits + 3
        __context.prec = digits
        e = Decimal(0)
        one = Decimal(1)
        factorial = one
        n = 0
        current = one
        eps = one / Decimal(10 ** digits)
        while current >= eps:
            e += current
            n += 1
            factorial *= n
            current = one / factorial
        __context.prec -= 2
        return +e

    if method == "series":
        return series_e()
    elif method == "spigot":
        return spigot_e()
    else:
        raise ValueError("Invalid method. Use 'series' or 'spigot'")


def calc_pi(digits: int, method: str = "chudnovsky") -> Decimal:
    def chudnovsky_pi():
        nonlocal digits
        digits = digits + 3
        __context.prec = digits
        c = 426880 * Decimal(10005).sqrt()
        m = 1
        _l = 13591409
        x = 1
        k = 6
        s = _l
        const = 14
        end = int(digits / const) + 1
        for i in range(1, end):
            m = (k ** 3 - 16 * k) * m // (i ** 3)
            _l += 545140134
            x *= -262537412640768000
            s += Decimal(m * _l) / x
            k += 12
        pi = c / s
        __context.prec -= 2
        return +pi

    def bbp_pi():
        nonlocal digits
        digits = digits + 3
        __context.prec = digits
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
            if k > digits:
                break
            p16 <<= 4
        __context.prec -= 2
        return +pi

    if method == "chudnovsky":
        return chudnovsky_pi()
    elif method == "bbp":
        return bbp_pi()
    else:
        raise ValueError("Invalid method. Use 'chudnovsky' or 'bbp'")


def calc_phi(digits: int, method: str = "algebraic") -> Decimal:
    def algebraic_phi():
        __context.prec = digits + 1
        one = Decimal(1)
        five = Decimal(5)
        two = Decimal(2)
        return (one + five.sqrt()) / two

    def newton_phi():
        nonlocal digits
        digits = digits + 3
        __context.prec = digits
        one = Decimal(1)
        two = Decimal(2)
        x = Decimal("1.5")
        eps = one / Decimal(10 ** digits)
        while True:
            x_new = x - (x ** 2 - x - one) / (two * x - one)
            if abs(x_new - x) < eps:
                break
            x = x_new
        __context.prec -= 2
        return +x

    if method == "algebraic":
        return algebraic_phi()
    elif method == "newton":
        return newton_phi()
    else:
        raise ValueError("Invalid method. Use 'algebraic' or 'newton'")
