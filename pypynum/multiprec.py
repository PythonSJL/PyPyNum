from decimal import Decimal, getcontext as __
from fractions import Fraction
from .types import Any, Union, prec

__context = __()
del __


def _setprec(sigfigs: int):
    __context.prec = sigfigs


def frac2dec(frac: Fraction, sigfigs: int) -> Decimal:
    __context.prec = sigfigs
    return Decimal(frac.numerator) / Decimal(frac.denominator)


def mp_e(sigfigs: int, method: str = "series") -> Decimal:
    def series_e():
        nonlocal sigfigs
        sigfigs += 5
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
        sigfigs += 5
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
        sigfigs += 5
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
        sigfigs += 5
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
    x = Decimal(str(x))
    sigfigs += 5
    __context.prec = sigfigs
    sin_x = Decimal(0)
    pi = mp_pi(sigfigs + 5)
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
    x = Decimal(str(x))
    sigfigs += 5
    __context.prec = sigfigs
    cos_x = Decimal(0)
    pi = mp_pi(sigfigs + 5)
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
        return Decimal(str(x)).ln()
    x = Decimal(str(x))
    if x <= 0:
        raise ValueError("Natural logarithm is not defined for x <= 0")
    sigfigs += 5
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
    while abs(term) >= eps:
        term *= dx
        ln_x += term / k
        k += 1
    __context.prec -= 5
    return ln_x * sign


def mp_log(x: prec, base: prec, sigfigs: int, builtin: bool = True) -> Decimal:
    sigfigs += 5
    if builtin:
        __context.prec = sigfigs
        x_dec = Decimal(str(x))
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


def mp_atan(x: prec, sigfigs: int) -> Decimal:
    x = Decimal(str(x))
    if abs(x) != x:
        return -mp_atan(-x, sigfigs)
    __context.prec = sigfigs + 5
    m = 0
    if x > 2:
        x = 1 / x
        flag = -1
    else:
        flag = 0
        if x > 0.5:
            x = x / (1 + (1 + x ** 2).sqrt())
            m = 2
            flag = 1
            if x > 0.5:
                x = x / (1 + (1 + x ** 2).sqrt())
                m = 4
    atan_x = 0
    term = power = x
    n = 1
    constant = -x * x
    eps = Decimal(10) ** -sigfigs
    while abs(term) >= eps:
        atan_x += term
        power *= constant
        n += 2
        term = power / n
    if flag:
        atan_x = atan_x * m if flag > 0 else mp_pi(sigfigs + 5) - atan_x
    atan_x %= mp_pi(sigfigs + 5) / 2
    __context.prec = sigfigs
    return +atan_x


def mp_atan2(y: prec, x: prec, sigfigs: int) -> Decimal:
    pi = mp_pi(sigfigs + 5)
    if x == 0:
        if y == 0:
            raise ValueError("Cannot compute atan2(0, 0)")
        if y > 0:
            return pi / 2
        else:
            return -pi / 2
    result = mp_atan(y / x, sigfigs + 5)
    if x < 0:
        if y >= 0:
            result += pi
        else:
            result -= pi
    __context.prec = sigfigs
    return +result


def mp_exp(x: prec, sigfigs: int, builtin: bool = True) -> Decimal:
    if builtin:
        __context.prec = sigfigs
        return Decimal(str(x)).exp()
    x = Decimal(str(x))
    __context.prec = sigfigs + 5
    n = term = exp_x = Decimal(1)
    eps = Decimal(10) ** -sigfigs
    while abs(term) >= eps:
        term *= x / n
        exp_x += term
        n += 1
    __context.prec = sigfigs
    return +exp_x


def mp_sinh(x: prec, sigfigs: int) -> Decimal:
    x = Decimal(str(x))
    sinh_x = (Decimal(1) - mp_exp(-2 * x, sigfigs + 5)) / (2 * mp_exp(-x, sigfigs + 5))
    __context.prec = sigfigs
    return +sinh_x


def mp_cosh(x: prec, sigfigs: int) -> Decimal:
    x = Decimal(str(x))
    cosh_x = (Decimal(1) + mp_exp(-2 * x, sigfigs + 5)) / (2 * mp_exp(-x, sigfigs + 5))
    __context.prec = sigfigs
    return +cosh_x


def mp_asin(x: prec, sigfigs: int) -> Decimal:  # FIXME
    __context.prec = sigfigs + 5
    x = Decimal(str(x))
    if x < -1 or x > 1:
        raise ValueError("Input must be within the domain of definition [-1, 1]")
    original_x = x
    if x > 0.5:
        x = (1 - x ** 2).sqrt()
    elif x < -0.5:
        x = -(1 - x ** 2).sqrt()
    asin_x = Decimal(0)
    x_squared = x * x
    numerator = x
    denominator = term = 1
    eps = Decimal(10) ** -sigfigs
    while abs(term) >= eps:
        term = numerator / denominator
        asin_x += term
        denominator += 2
        numerator *= x_squared * (denominator - 2) / (denominator - 1)
    if original_x > 0.5:
        asin_x = mp_pi(sigfigs + 5) / 2 - asin_x
    elif original_x < -0.5:
        asin_x = -mp_pi(sigfigs + 5) / 2 - asin_x
    __context.prec = sigfigs
    return +asin_x


def mp_acos(x: prec, sigfigs: int) -> Decimal:  # FIXME
    x = Decimal(str(x))
    if x < -1 or x > 1:
        raise ValueError("Input must be within the domain of definition [-1, 1]")
    acos_x = mp_pi(sigfigs + 5) / 2 - mp_asin(x, sigfigs + 5)
    __context.prec = sigfigs
    return +acos_x


def mp_fresnel_s(x: prec, sigfigs: int) -> Decimal:  # FIXME
    x = Decimal(str(x))
    more = int(x * x)
    sigfigs += more
    __context.prec = sigfigs
    s_x = Decimal(0)
    p = mp_pi(sigfigs + 5)
    f = -x ** 4 * p ** 2 / 4
    num = p / 2
    coeff = 3
    fact = 1
    term = num / coeff
    n = 0
    eps = Decimal(10) ** -sigfigs
    while abs(term) >= eps:
        s_x += term
        n += 1
        num *= f
        coeff += 4
        fact *= (2 * n) * (2 * n + 1)
        term = num / (coeff * fact)
    s_x *= x ** 3
    __context.prec -= more
    return +s_x


def mp_fresnel_c(x: prec, sigfigs: int) -> Decimal:  # FIXME
    x = Decimal(str(x))
    more = int(x * x)
    sigfigs += more
    __context.prec = sigfigs
    c_x = Decimal(0)
    f = -x ** 4 * mp_pi(sigfigs + 5) ** 2 / 4
    num = Decimal(1)
    fact_2k = 1
    k = 0
    term = num / fact_2k
    n = 0
    eps = Decimal(10) ** -sigfigs
    while abs(term) >= eps:
        c_x += term
        n += 1
        num *= f
        fact_2k *= (2 * n) * (2 * n - 1)
        k += 4
        term = num / (fact_2k + k * fact_2k)
    c_x *= x
    __context.prec -= more
    return +c_x


def mp_euler_gamma(sigfigs: int) -> Decimal:
    sigfigs += 5
    __context.prec = sigfigs
    n = int(sigfigs * 0.6)
    i_n = Decimal(0)
    j_n = Decimal(0)
    k = 1
    n_power_k = n
    fact_k = Decimal(1)
    harmonic_sum = Decimal(0)
    term = Decimal(1)
    eps = Decimal(10) ** -sigfigs
    while abs(term) >= eps:
        term = (n_power_k / fact_k) ** 2
        i_n += term * harmonic_sum
        j_n += term
        n_power_k *= n
        fact_k *= k
        harmonic_sum += Decimal(1) / k
        k += 1
    gamma = i_n / j_n - Decimal(n).ln()
    __context.prec -= 5
    return +gamma


class MPComplex:
    def __init__(self, real, imag, sigfigs=28):
        _setprec(sigfigs)
        self.real = +Decimal(str(real))
        self.imag = +Decimal(str(imag))
        self.sigfigs = sigfigs

    def __add__(self, other):
        other = asmpc(other, sigfigs=self.sigfigs)
        return MPComplex(self.real + other.real, self.imag + other.imag, self.sigfigs)

    def __sub__(self, other):
        other = asmpc(other, sigfigs=self.sigfigs)
        return MPComplex(self.real - other.real, self.imag - other.imag, self.sigfigs)

    def __mul__(self, other):
        other = asmpc(other, sigfigs=self.sigfigs)
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag + self.imag * other.real
        return MPComplex(real, imag, self.sigfigs)

    def __truediv__(self, other):
        other = asmpc(other, sigfigs=self.sigfigs)
        if other.real == 0 and other.imag == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        else:
            denominator = other.real ** 2 + other.imag ** 2
            real = (self.real * other.real + self.imag * other.imag) / denominator
            imag = (self.imag * other.real - self.real * other.imag) / denominator
            return MPComplex(real, imag, self.sigfigs)

    def __neg__(self):
        return MPComplex(-self.real, -self.imag, self.sigfigs)

    __radd__ = __add__
    __rmul__ = __mul__

    def conjugate(self):
        return MPComplex(self.real, -self.imag, self.sigfigs)

    def modulus(self, sigfigs=None):
        if sigfigs is not None:
            _setprec(sigfigs)
        mod = (self.real ** 2 + self.imag ** 2).sqrt()
        _setprec(self.sigfigs)
        return +mod

    def __abs__(self):
        return self.modulus()

    def __pow__(self, power, modulo=None):
        if modulo is not None:
            raise TypeError("Modulo operation is not supported for complex numbers")
        if isinstance(power, float):
            power = Decimal(str(power))
        elif isinstance(power, complex):
            power = MPComplex(str(power.real), str(power.imag), self.sigfigs)
        modulus = self.modulus(self.sigfigs + 5)
        angle = self.argument(self.sigfigs + 5)
        if isinstance(power, (int, Decimal)):
            new_modulus = modulus ** power
            new_angle = angle * power
        elif isinstance(power, MPComplex):
            ln_modulus = mp_ln(modulus, self.sigfigs + 5)
            exp_real = power.real
            exp_imag = power.imag
            new_modulus = (ln_modulus * exp_real - exp_imag * angle).exp()
            new_angle = exp_imag * ln_modulus + exp_real * angle
        else:
            raise ValueError("Exponent must be a real number, Decimal, or MPComplex instance")
        real = new_modulus * mp_cos(new_angle, self.sigfigs + 5)
        imag = new_modulus * mp_sin(new_angle, self.sigfigs + 5)
        return MPComplex(real, imag, self.sigfigs)

    def sqrt(self):
        modulus = self.modulus(self.sigfigs + 5)
        angle = self.argument(self.sigfigs + 5)
        sqrt_modulus = modulus.sqrt()
        half_angle = angle / 2
        real = sqrt_modulus * mp_cos(half_angle, self.sigfigs + 5)
        imag = sqrt_modulus * mp_sin(half_angle, self.sigfigs + 5)
        return MPComplex(real, imag, self.sigfigs)

    def cbrt(self):
        modulus = self.modulus(self.sigfigs + 5)
        angle = self.argument(self.sigfigs + 5)
        cbrt_modulus = modulus ** (Decimal(1) / 3)
        third_angle = angle / 3
        real = cbrt_modulus * mp_cos(third_angle, self.sigfigs + 5)
        imag = cbrt_modulus * mp_sin(third_angle, self.sigfigs + 5)
        return MPComplex(real, imag, self.sigfigs)

    def argument(self, sigfigs=None):
        if sigfigs is None:
            sigfigs = self.sigfigs
        return mp_atan2(self.imag, self.real, sigfigs)

    def ln(self):
        modulus = self.modulus(self.sigfigs + 5)
        angle = self.argument()
        return MPComplex(mp_ln(modulus, self.sigfigs), angle, self.sigfigs)

    def exp(self):
        e_real = self.real.exp()
        real_part = e_real * mp_cos(self.imag, self.sigfigs)
        imag_part = e_real * mp_sin(self.imag, self.sigfigs)
        return MPComplex(real_part, imag_part, self.sigfigs)

    def __str__(self):
        real_str = _remove_trailing_zeros(self.real)
        imag_str = _remove_trailing_zeros(self.imag)
        complex_str = ""
        if real_str != "0":
            complex_str += real_str
        if imag_str != "0":
            if (complex_str and imag_str[0] != "-") or (not complex_str and imag_str[0] == "-"):
                imag_sign = "+" if imag_str[0] != "-" else ""
            else:
                imag_sign = "-" if imag_str[0] == "-" else ""
                imag_str = imag_str.lstrip("+-")
            complex_str += imag_sign + imag_str + "i"
        if not complex_str:
            complex_str = "0"
        return complex_str

    def __repr__(self):
        return "MPComplex({!r}, {!r}, sigfigs={!r})".format(self.real, self.imag, self.sigfigs)

    def sin(self):
        real_part = mp_sin(self.real, self.sigfigs + 5) * mp_cosh(self.imag, self.sigfigs + 5)
        imag_part = mp_cos(self.real, self.sigfigs + 5) * mp_sinh(self.imag, self.sigfigs + 5)
        return MPComplex(real_part, imag_part, self.sigfigs)

    def cos(self):
        real_part = mp_cos(self.real, self.sigfigs + 5) * mp_cosh(self.imag, self.sigfigs + 5)
        imag_part = -mp_sin(self.real, self.sigfigs + 5) * mp_sinh(self.imag, self.sigfigs + 5)
        return MPComplex(real_part, imag_part, self.sigfigs)

    def tan(self):
        cos_z = self.cos()
        if cos_z.real == 0 and cos_z.imag == 0:
            raise ZeroDivisionError("Cosine of the complex number is zero, cannot compute tangent")
        return self.sin() / cos_z

    def cot(self):
        sin_z = self.sin()
        if sin_z.real == 0 and sin_z.imag == 0:
            raise ZeroDivisionError("Sine of the complex number is zero, cannot compute cotangent")
        return self.cos() / sin_z

    def sec(self):
        cos_z = self.cos()
        if cos_z.real == 0 and cos_z.imag == 0:
            raise ZeroDivisionError("Cosine of the complex number is zero, cannot compute secant")
        return MPComplex(1, 0, self.sigfigs) / cos_z

    def csc(self):
        sin_z = self.sin()
        if sin_z.real == 0 and sin_z.imag == 0:
            raise ZeroDivisionError("Sine of the complex number is zero, cannot compute cosecant")
        return MPComplex(1, 0, self.sigfigs) / sin_z


def _remove_trailing_zeros(value: Any) -> str:
    value_str = str(value).lower()
    if value_str == "0" or value_str.startswith(("0e", "-0e")):
        return "0"
    if "." not in value_str:
        return value_str
    if "e" in value_str:
        mantissa, exponent = value_str.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return mantissa + "e" + exponent
    return value_str.rstrip("0").rstrip(".")


def asmpc(real: Union[int, float, str, Decimal, complex, MPComplex], imag: prec = 0,
          sigfigs: int = 28) -> MPComplex:
    if isinstance(real, (complex, MPComplex)):
        if imag != 0:
            raise ValueError("When 'real' is a complex number, 'imag' must be 0")
        return MPComplex(real.real, real.imag, sigfigs)
    elif isinstance(real, (int, float, str, Decimal)):
        if not isinstance(imag, (int, float, str, Decimal)):
            raise TypeError("When 'real' is an int, float, str, or Decimal, 'imag' must also be an int, float, str, "
                            "or Decimal")
        return MPComplex(real, imag, sigfigs)
    else:
        raise TypeError("Cannot convert to MPComplex: 'real' must be a complex number, int, float, str, or Decimal")
