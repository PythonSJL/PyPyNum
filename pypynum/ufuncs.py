from cmath import log as _clog, exp as _cexp
from math import (lgamma as _loggamma, gamma as _gamma, pi as _pi, ceil as _ceil,
                  log as _log, inf as _inf, sin as _sin, exp as _exp)
from .arrays import Array, BoolArray


def apply(a, func, rtype=None, dtype=None):
    def inner(data):
        return list(map(inner, data)) if isinstance(data, list) else func(data)

    if isinstance(a, Array):
        if callable(func):
            if rtype is None:
                rtype = type(a)
            if dtype is None:
                dtype = a.dtype
            result_data = inner(a.data)
            if issubclass(rtype, Array):
                return rtype(result_data, dtype=dtype)
            return rtype(result_data)
        else:
            raise TypeError("The 'func' argument must be callable, got {} instead".format(type(func)))
    raise TypeError("The 'a' argument must be an instance of Array, got {} instead".format(type(a)))


def base_ufunc(*arrays, func, args=(), rtype=None, dtype=None):
    if not all([isinstance(item, Array) for item in arrays]) or not callable(func):
        raise TypeError('The input parameter type is incorrect')
    if len(set([item.shape for item in arrays])) != 1:
        raise ValueError('All arrays must have the same shape')
    data = [item.data for item in arrays]
    if args:
        def inner(arrs):
            if isinstance(arrs[0], list):
                return list(map(inner, zip(*arrs)))
            return func(*arrs, *args)
    else:
        def inner(arrs):
            if isinstance(arrs[0], list):
                return list(map(inner, zip(*arrs)))
            return func(*arrs)
    if rtype is None:
        rtype = type(arrays[0])
    if dtype is None:
        dtype = arrays[0].dtype
    result_data = inner(data)
    if issubclass(rtype, Array):
        return rtype(result_data, dtype=dtype)
    return rtype(result_data)


def ufunc_helper(x, y, func, rtype=None, dtype=None):
    if isinstance(x, Array):
        if isinstance(y, Array):
            return base_ufunc(x, y, func=func, rtype=rtype, dtype=dtype)
        return base_ufunc(x, func=func, args=[y], rtype=rtype, dtype=dtype)
    elif isinstance(y, Array):
        return base_ufunc(y, func=lambda b, a: func(a, b), args=[x], rtype=rtype, dtype=dtype)
    else:
        return func(x, y)


def add(x, y):
    return ufunc_helper(x, y, lambda a, b: a + b)


def subtract(x, y):
    return ufunc_helper(x, y, lambda a, b: a - b)


def multiply(x, y):
    return ufunc_helper(x, y, lambda a, b: a * b)


def divide(x, y):
    return ufunc_helper(x, y, lambda a, b: a / b)


def floor_divide(x, y):
    return ufunc_helper(x, y, lambda a, b: a // b)


def modulo(x, y):
    return ufunc_helper(x, y, lambda a, b: a % b)


def power(x, y, m=None):
    from .arrays import full
    def inner(_x, _y, _m=None):
        try:
            return pow(_x, _y, _m)
        except ValueError:
            return pow(_x, _y)

    is_array = (isinstance(x, Array), isinstance(y, Array), isinstance(m, Array))
    if not any(is_array):
        return inner(x, y, m)
    first = (x, y, m)[is_array.index(True)]
    shape = first.shape
    if not is_array[0]:
        x = full(shape, x)
    if not is_array[1]:
        y = full(shape, y)
    if not is_array[2]:
        m = full(shape, m)
    return base_ufunc(x, y, m, func=inner, rtype=type(first))


def greater_than(x, y):
    return ufunc_helper(x, y, lambda a, b: a > b, rtype=BoolArray)


def greater_equal(x, y):
    return ufunc_helper(x, y, lambda a, b: a >= b, rtype=BoolArray)


def equal(x, y):
    return ufunc_helper(x, y, lambda a, b: a == b, rtype=BoolArray)


def less_equal(x, y):
    return ufunc_helper(x, y, lambda a, b: a <= b, rtype=BoolArray)


def less_than(x, y):
    return ufunc_helper(x, y, lambda a, b: a < b, rtype=BoolArray)


def not_equal(x, y):
    return ufunc_helper(x, y, lambda a, b: a != b, rtype=BoolArray)


_G = 607.0 / 128.0
_C0 = 0.9999999999999971
_C_COEFFS = (57.15623566586292, -59.59796035547549, 14.136097974741746, -0.4919138160976202, 3.399464998481189e-05,
             4.652362892704858e-05, -9.837447530487956e-05, 0.0001580887032249125, -0.00021026444172410488,
             0.00021743961811521265, -0.0001643181065367639, 8.441822398385275e-05, -2.6190838401581408e-05,
             3.6899182659531625e-06)
_LOG_2PI = _log(2.0 * _pi) * 0.5
_LOG_2 = _log(2.0)


def _lanczos_sum(z):
    return _C0 + sum((c / (z + k) for k, c in enumerate(_C_COEFFS)))


def _log_sin_pi(z):
    if z.imag > 0:
        w = _cexp(2j * _pi * z)
        return -_LOG_2 + 1j * _pi * (0.5 - z) + _clog(1 - w)
    if z.imag < 0:
        w = _cexp(-2j * _pi * z)
        return -_LOG_2 + 1j * _pi * (z - 0.5) + _clog(1 - w)
    s = _sin(_pi * z.real)
    if s == 0:
        raise ValueError('Pole at sin(π·{})'.format(z))
    return _log(abs(s)) + 1j * _pi * _ceil(-z.real)


def _scalar_loggamma(z):
    if isinstance(z, complex):
        if z.imag == 0.0 and z.real <= 0 and float(z.real).is_integer():
            raise ValueError('Pole at {}'.format(z))
    elif z <= 0 and float(z).is_integer():
        raise ValueError('Pole at {}'.format(z))
    if isinstance(z, complex):
        if z.real < 0.5:
            return _clog(_pi) - _log_sin_pi(z) - _scalar_loggamma(1 - z)
        shift = 0
        while z.real < 10.0:
            z += 1
            shift += 1
        zh, zgh = (z - 0.5, z + _G - 0.5)
        result = zh * _clog(zgh) - zgh + _LOG_2PI + _clog(_lanczos_sum(z))
        for _ in range(shift):
            z -= 1
            result -= _clog(z)
        return result
    return _loggamma(z)


def _scalar_gamma(z):
    if isinstance(z, complex):
        if z.imag == 0.0 and z.real <= 0 and float(z.real).is_integer():
            raise ValueError('Pole at {}'.format(int(z.real)))
    elif z <= 0 and float(z).is_integer():
        raise ValueError('Pole at {}'.format(int(z)))
    if isinstance(z, complex):
        return _cexp(_scalar_loggamma(z))
    try:
        return _gamma(z)
    except OverflowError:
        return _inf


def loggamma(z):
    if isinstance(z, Array):
        return apply(z, _scalar_loggamma)
    return _scalar_loggamma(z)


def gamma(z):
    if isinstance(z, Array):
        return apply(z, _scalar_gamma)
    return _scalar_gamma(z)


def _beta_scalar(a, b):
    log_val = _logbeta_scalar(a, b)
    if isinstance(log_val, complex):
        return _cexp(log_val)
    try:
        return _exp(log_val)
    except OverflowError:
        return _inf


def beta(a, b):
    return ufunc_helper(a, b, _beta_scalar)


def _logbeta_scalar(a, b):
    return _scalar_loggamma(a) + _scalar_loggamma(b) - _scalar_loggamma(a + b)


def logbeta(a, b):
    return ufunc_helper(a, b, _logbeta_scalar)


def _factorial_scalar(n):
    return _scalar_gamma(n + 1)


def factorial(n):
    if isinstance(n, Array):
        return apply(n, _factorial_scalar)
    return _factorial_scalar(n)


def _logfactorial_scalar(n):
    return _scalar_loggamma(n + 1)


def logfactorial(n):
    if isinstance(n, Array):
        return apply(n, _logfactorial_scalar)
    return _logfactorial_scalar(n)


def _pochhammer_scalar(a, n):
    return _scalar_gamma(a + n) / _scalar_gamma(a)


def pochhammer(a, n):
    return ufunc_helper(a, n, _pochhammer_scalar)


def _binomial_scalar(n, k):
    log_val = _scalar_loggamma(n + 1) - _scalar_loggamma(k + 1) - _scalar_loggamma(n - k + 1)
    if isinstance(log_val, complex):
        return _cexp(log_val)
    try:
        return _exp(log_val)
    except OverflowError:
        return _inf


def binomial(n, k):
    return ufunc_helper(n, k, _binomial_scalar)
