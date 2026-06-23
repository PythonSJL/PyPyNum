from math import cos as __cos, sin as __sin
from random import randint as __randint, random as __random, uniform as __uniform
from .types import real as __real


def randint_polar(left: int, right: int, mod: __real = None, angle: __real = None) -> complex:
    """
    Generate a complex number in polar form with integer modulus and angle
    randomly selected from a specified range.
    :param left: The lower bound (inclusive) of the random number range.
    :param right: The upper bound (inclusive) of the random number range.
    :param mod: The modulus of the complex number. If None, it is randomly
                selected from the range [left, right].
    :param angle: The angle of the complex number. If None, it is randomly
                  selected from the range [0, 2π].
    :return: A complex number in polar form.
    """
    mod = __randint(left, right) if mod is None else mod
    angle = __uniform(0, 6.283185307179586) if angle is None else angle
    return mod * (__cos(angle) + __sin(angle) * 1j)


def randint_rect(left: int, right: int, real: __real = None, imag: __real = None) -> complex:
    """
    Generate a complex number in rectangular form with integer real and
    imaginary parts randomly selected from a specified range.
    :param left: The lower bound (inclusive) of the random number range.
    :param right: The upper bound (inclusive) of the random number range.
    :param real: The real part of the complex number. If None, it is randomly
                 selected from the range [left, right].
    :param imag: The imaginary part of the complex number. If None, it is
                 randomly selected from the range [left, right].
    :return: A complex number in rectangular form.
    """
    real = __randint(left, right) if real is None else real
    imag = __randint(left, right) if imag is None else imag
    return complex(real, imag)


def random_polar(mod: __real = None, angle: __real = None) -> complex:
    """
    Generate a complex number in polar form with floating-point modulus and
    angle randomly selected from the range [0, 2π].
    :param mod: The modulus of the complex number. If None, it is randomly
                selected from the range [0, 1].
    :param angle: The angle of the complex number. If None, it is uniformly
                  selected from the range [0, 2π].
    :return: A complex number in polar form.
    """
    mod = __random() if mod is None else mod
    angle = __uniform(0, 6.283185307179586) if angle is None else angle
    return mod * (__cos(angle) + __sin(angle) * 1j)


def random_rect(real: __real = None, imag: __real = None) -> complex:
    """
    Generate a complex number in rectangular form with floating-point real and
    imaginary parts randomly selected from the range [0, 1].
    :param real: The real part of the complex number. If None, it is randomly
                 selected from the range [0, 1].
    :param imag: The imaginary part of the complex number. If None, it is
                 randomly selected from the range [0, 1].
    :return: A complex number in rectangular form.
    """
    real = __random() if real is None else real
    imag = __random() if imag is None else imag
    return complex(real, imag)


def uniform_polar(left: __real, right: __real, mod: __real = None, angle: __real = None) -> complex:
    """
    Generate a complex number in polar form with uniformly distributed
    floating-point modulus and angle from a specified range.
    :param left: The lower bound (inclusive) of the random number range.
    :param right: The upper bound (inclusive) of the random number range.
    :param mod: The modulus of the complex number. If None, it is uniformly
                selected from the range [left, right].
    :param angle: The angle of the complex number. If None, it is uniformly
                  selected from the range [0, 2π].
    :return: A complex number in polar form.
    """
    mod = __uniform(left, right) if mod is None else mod
    angle = __uniform(0, 6.283185307179586) if angle is None else angle
    return mod * (__cos(angle) + __sin(angle) * 1j)


def uniform_rect(left: __real, right: __real, real: __real = None, imag: __real = None) -> complex:
    """
    Generate a complex number in rectangular form with uniformly distributed
    floating-point real and imaginary parts from a specified range.
    :param left: The lower bound (inclusive) of the random number range.
    :param right: The upper bound (inclusive) of the random number range.
    :param real: The real part of the complex number. If None, it is uniformly
                 selected from the range [left, right].
    :param imag: The imaginary part of the complex number. If None, it is
                 uniformly selected from the range [left, right].
    :return: A complex number in rectangular form.
    """
    real = __uniform(left, right) if real is None else real
    imag = __uniform(left, right) if imag is None else imag
    return complex(real, imag)
