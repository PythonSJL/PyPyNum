from .tools import linspace
from .types import arr


def interp1d(data: arr, length: int) -> list:
    """
    Introduction
    ==========
    One-dimensional data interpolation.

    Example
    ==========
    >>> interp1d((2, 4, 4, 2), 6)
    [2, 3.320000000000001, 4.160000000000005, 4.160000000000012, 3.3200000000000074, 2]
    >>>
    :param data: List of data points to be interpolated. Must be at least two points.
    :param length: The number of points in the resampled data set.
    :return: A list of interpolated values at the new points.
    """
    from .regression import lin_reg, par_reg
    expr = [lambda x: sum([k * x ** (1 - n) for n, k in enumerate(lin_reg([0, 1], [data[0], data[1]]))])]
    for i in range(len(data) - 2):
        temp = par_reg(list(range(i, i + 3)), data[i:i + 3])
        expr.append(lambda x, coefficients=temp: sum([k * x ** (2 - n) for n, k in enumerate(coefficients)]))
    expr.append(lambda x: sum([k * x ** (1 - n) for n, k in
                               enumerate(lin_reg([len(data) - 2, len(data) - 1], [data[-2], data[-1]]))]))
    result = linspace(0, len(data) - 1, length)
    for item in range(length):
        if int(result[item]) != result[item]:
            result[item] = (expr[int(result[item])](result[item]) + expr[int(result[item] + 1)](result[item])) / 2
        else:
            result[item] = data[int(result[item])]
    return result


def bicubic(x):
    """
    Calculate the cubic B-spline interpolation function value.
    :param x: The x value for which the B-spline function is evaluated.
    :return: The value of the cubic B-spline function at x.
    """
    absx = abs(x)
    if absx <= 1:
        return 1.5 * absx ** 3 - 2.5 * absx ** 2 + 1
    elif absx < 2:
        return -0.5 * absx ** 3 + 2.5 * absx ** 2 - 4 * absx + 2
    else:
        return 0


def contribute(src, x, y, channels=None):
    """
    Calculate the contribution of the source array at a specific point after bicubic interpolation.
    :param src: The source 2D array from which to interpolate.
    :param x: The x-coordinate of the point to interpolate.
    :param y: The y-coordinate of the point to interpolate.
    :param channels: The number of channels if src is a multichannel array.
    :return: The interpolated value at the point (x, y).
    """
    src_height = len(src)
    src_width = len(src[0])
    value = 0.0 if channels is None else [0.0] * channels
    x_int = int(x)
    y_int = int(y)
    dx = x - x_int
    dy = y - y_int
    for j in range(-1, 3):
        ny = y_int + j
        if 0 <= ny < src_height:
            wy = bicubic(j - dy)
            for i in range(-1, 3):
                nx = x_int + i
                if 0 <= nx < src_width:
                    wx = bicubic(i - dx)
                    if channels is None:
                        value += wx * wy * src[ny][nx]
                    else:
                        for c in range(channels):
                            value[c] += wx * wy * src[ny][nx][c]
    return value


def interp2d(src, new_height, new_width, channels=None, round_res=False, min_val=None, max_val=None):
    """
    Introduction
    ==========
    Two-dimensional data interpolation using bicubic spline interpolation.

    Example
    ==========
    >>> interp2d([[1, 2], [3, 4]], 3, 3)
    [[1.0, 1.6875, 2.0], [2.25, 3.1640625, 3.375], [3.0, 3.9375, 4.0]]
    >>>
    :param src: The source 2D array to be interpolated.
    :param new_height: The desired height of the interpolated array.
    :param new_width: The desired width of the interpolated array.
    :param channels: The number of channels if src is a multichannel array.
    :param round_res: Whether to round the result to the nearest integer.
    :param min_val: The minimum value to clip the interpolated results.
    :param max_val: The maximum value to clip the interpolated results.
    :return: A 2D array of the interpolated values with the new dimensions.
    """
    src_height = len(src)
    src_width = len(src[0])
    dst = [[None] * new_width for _ in range(new_height)]
    for dst_y in range(new_height):
        for dst_x in range(new_width):
            src_x = (src_width - 1) * dst_x / (new_width - 1)
            src_y = (src_height - 1) * dst_y / (new_height - 1)
            value = contribute(src, src_x, src_y, channels)
            if min_val is not None:
                value = max(min_val, value) if channels is None else tuple(max(min_val, v) for v in value)
            if max_val is not None:
                value = min(max_val, value) if channels is None else tuple(min(max_val, v) for v in value)
            if round_res:
                value = round(value) if channels is None else tuple(map(round, value))
            dst[dst_y][dst_x] = value
    return dst
