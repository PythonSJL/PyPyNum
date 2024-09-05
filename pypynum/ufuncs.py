from .Array import Array as __Array


def apply(a, func, rtype=None):
    def inner(data):
        return [inner(item) for item in data] if isinstance(data, list) else func(data)

    if isinstance(a, __Array):
        if callable(func):
            if rtype is None:
                rtype = type(a)
            return rtype(inner(a.data))
        else:
            raise TypeError("The 'func' argument must be callable, got {} instead".format(type(func)))
    else:
        raise TypeError("The 'a' argument must be an instance of Array, got {} instead".format(type(a)))


def base_ufunc(*arrays, func, args=(), rtype=None):
    _type = str(type(func))
    if not all([isinstance(item, __Array) for item in arrays]) or not (
            _type.startswith("<function ") or _type.startswith("<class ")):
        raise TypeError("The input parameter type is incorrect")
    if len(set([item.shape for item in arrays])) != 1:
        raise ValueError("All arrays must have the same shape")
    data = [item.data for item in arrays]

    def inner(*arrs):
        if isinstance(arrs[0], list):
            _copy = []
            for item in zip(*arrs):
                _copy.append(inner(*item))
            return _copy
        else:
            return func(*arrs, *args) if args else func(*arrs)

    if rtype is None:
        rtype = type(arrays[0])
    return rtype(inner(*data))


def ufunc_helper(x, y, func):
    if isinstance(x, __Array):
        if isinstance(y, __Array):
            return base_ufunc(x, y, func=func)
        else:
            return base_ufunc(x, func=func, args=[y])
    else:
        if isinstance(y, __Array):
            return base_ufunc(y, func=lambda b, a: func(a, b), args=[x])
        else:
            return func(x, y)


def add(x, y):
    def inner(_x, _y):
        return _x + _y

    return ufunc_helper(x, y, inner)


def subtract(x, y):
    def inner(_x, _y):
        return _x - _y

    return ufunc_helper(x, y, inner)


def multiply(x, y):
    def inner(_x, _y):
        return _x * _y

    return ufunc_helper(x, y, inner)


def divide(x, y):
    def inner(_x, _y):
        return _x / _y

    return ufunc_helper(x, y, inner)


def floor_divide(x, y):
    def inner(_x, _y):
        return _x // _y

    return ufunc_helper(x, y, inner)


def modulo(x, y):
    def inner(_x, _y):
        return _x % _y

    return ufunc_helper(x, y, inner)


def power(x, y, m=None):
    from .Array import full

    def inner(_x, _y, _m=None):
        try:
            return pow(_x, _y, _m)
        except ValueError:
            return pow(_x, _y)

    is_array = [isinstance(x, __Array), isinstance(y, __Array), isinstance(m, __Array)]
    if not any(is_array):
        return pow(x, y, m)
    first = (x, y, m)[is_array.index(True)]
    shape = first.shape
    if not is_array[0]:
        x = full(shape, x)
    if not is_array[1]:
        y = full(shape, y)
    if not is_array[2]:
        m = full(shape, m)
    return base_ufunc(x, y, m, func=inner, rtype=type(first))


def gt(x, y):
    return ufunc_helper(x, y, lambda a, b: a > b)


def ge(x, y):
    return ufunc_helper(x, y, lambda a, b: a >= b)


def eq(x, y):
    return ufunc_helper(x, y, lambda a, b: a == b)


def le(x, y):
    return ufunc_helper(x, y, lambda a, b: a <= b)


def lt(x, y):
    return ufunc_helper(x, y, lambda a, b: a < b)


def ne(x, y):
    return ufunc_helper(x, y, lambda a, b: a != b)
