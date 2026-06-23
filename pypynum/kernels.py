from .types import arr, num


def matmul2x2kernel(a: arr, b: arr) -> list:
    """
    Introduction
    ==========
    This function performs matrix multiplication for 2x2 matrices.

    Example
    ==========
    >>> matmul2x2kernel([[1, 2], [3, 4]], [[2, 0], [1, 2]])
    [[4, 4], [10, 8]]
    >>>
    :param a: The first 2x2 matrix to multiply.
    :param b: The second 2x2 matrix to multiply.
    :return: The result of the multiplication as a 2x2 matrix.
    """
    (a00, a01), (a10, a11) = a
    (b00, b01), (b10, b11) = b
    return [[a00 * b00 + a01 * b10, a00 * b01 + a01 * b11], [a10 * b00 + a11 * b10, a10 * b01 + a11 * b11]]


def matmul3x3kernel(a: arr, b: arr) -> list:
    """
    Introduction
    ==========
    This function performs matrix multiplication for 3x3 matrices.

    Example
    ==========
    >>> matmul3x3kernel([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    [[30, 24, 18], [84, 69, 54], [138, 114, 90]]
    >>>
    :param a: The first 3x3 matrix to multiply.
    :param b: The second 3x3 matrix to multiply.
    :return: The result of the multiplication as a 3x3 matrix.
    """
    (a00, a01, a02), (a10, a11, a12), (a20, a21, a22) = a
    (b00, b01, b02), (b10, b11, b12), (b20, b21, b22) = b
    return [[a00 * b00 + a01 * b10 + a02 * b20, a00 * b01 + a01 * b11 + a02 * b21, a00 * b02 + a01 * b12 + a02 * b22],
            [a10 * b00 + a11 * b10 + a12 * b20, a10 * b01 + a11 * b11 + a12 * b21, a10 * b02 + a11 * b12 + a12 * b22],
            [a20 * b00 + a21 * b10 + a22 * b20, a20 * b01 + a21 * b11 + a22 * b21, a20 * b02 + a21 * b12 + a22 * b22]]


def matmul4x4kernel(a: arr, b: arr) -> list:
    """
    Introduction
    ==========
    This function performs matrix multiplication for 4x4 matrices.

    Example
    ==========
    >>> matmul4x4kernel([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
    ... [[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]])
    [[80, 70, 60, 50], [240, 214, 188, 162], [400, 358, 316, 274], [560, 502, 444, 386]]
    >>>
    :param a: The first 4x4 matrix to multiply.
    :param b: The second 4x4 matrix to multiply.
    :return: The result of the multiplication as a 4x4 matrix.
    """
    (a00, a01, a02, a03), (a10, a11, a12, a13), (a20, a21, a22, a23), (a30, a31, a32, a33) = a
    (b00, b01, b02, b03), (b10, b11, b12, b13), (b20, b21, b22, b23), (b30, b31, b32, b33) = b
    return [[a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30, a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31,
             a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32, a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33],
            [a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30, a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31,
             a10 * b02 + a11 * b12 + a12 * b22 + a13 * b32, a10 * b03 + a11 * b13 + a12 * b23 + a13 * b33],
            [a20 * b00 + a21 * b10 + a22 * b20 + a23 * b30, a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31,
             a20 * b02 + a21 * b12 + a22 * b22 + a23 * b32, a20 * b03 + a21 * b13 + a22 * b23 + a23 * b33],
            [a30 * b00 + a31 * b10 + a32 * b20 + a33 * b30, a30 * b01 + a31 * b11 + a32 * b21 + a33 * b31,
             a30 * b02 + a31 * b12 + a32 * b22 + a33 * b32, a30 * b03 + a31 * b13 + a32 * b23 + a33 * b33]]


def det2x2kernel(a: arr) -> float:
    """
    Introduction
    ==========
    This function calculates the determinant of a 2x2 matrix.

    Example
    ==========
    >>> det2x2kernel([[1, 2], [3, 4]])
    -2
    >>>
    :param a: The 2x2 matrix to calculate the determinant of.
    :return: The determinant of the matrix.
    """
    (a00, a01), (a10, a11) = a
    return a00 * a11 - a01 * a10


def det3x3kernel(a: arr) -> float:
    """
    Introduction
    ==========
    This function calculates the determinant of a 3x3 matrix.

    Example
    ==========
    >>> det3x3kernel([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    0
    >>>
    :param a: The 3x3 matrix to calculate the determinant of.
    :return: The determinant of the matrix.
    """
    (a00, a01, a02), (a10, a11, a12), (a20, a21, a22) = a
    return a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)


def det4x4kernel(a: arr) -> float:
    """
    Introduction
    ==========
    This function calculates the determinant of a 4x4 matrix.

    Example
    ==========
    >>> det4x4kernel([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    0
    >>>
    :param a: The 4x4 matrix to calculate the determinant of.
    :return: The determinant of the matrix.
    """
    (a00, a01, a02, a03), (a10, a11, a12, a13), (a20, a21, a22, a23), (a30, a31, a32, a33) = a
    return a03 * (a12 * (a21 * a30 - a20 * a31) + a11 * (a20 * a32 - a22 * a30)) + a13 * (
            a02 * (a20 * a31 - a21 * a30) + a00 * (a21 * a32 - a22 * a31)) + a00 * a12 * (
            a23 * a31 - a21 * a33) + a01 * (a13 * (a22 * a30 - a20 * a32) + a12 * (a20 * a33 - a23 * a30)) + a10 * (
            a03 * (a22 * a31 - a21 * a32) + a02 * (a21 * a33 - a23 * a31) + a01 * (a23 * a32 - a22 * a33)) + a11 * (
            a02 * (a23 * a30 - a20 * a33) + a00 * (a22 * a33 - a23 * a32))


def inv2x2kernel(a: arr) -> list:
    """
    Introduction
    ==========
    This function calculates the inverse of a 2x2 matrix.

    Example
    ==========
    >>> inv2x2kernel([[1, 2], [3, 4]])
    [[-2.0, 1.0], [1.5, -0.5]]
    >>>
    :param a: The 2x2 matrix to calculate the inverse of.
    :return: The inverse of the matrix as a 2x2 matrix.
    """
    (a00, a01), (a10, a11) = a
    det = a00 * a11 - a01 * a10
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    return [[a11 / det, -a01 / det], [-a10 / det, a00 / det]]


def inv3x3kernel(a: arr) -> list:
    """
    Introduction
    ==========
    This function calculates the inverse of a 3x3 matrix.

    Example
    ==========
    >>> inv3x3kernel([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
    [[-24.0, 18.0, 5.0], [20.0, -15.0, -4.0], [-5.0, 4.0, 1.0]]
    >>>
    :param a: The 3x3 matrix to calculate the inverse of.
    :return: The inverse of the matrix as a 3x3 matrix.
    """
    (a00, a01, a02), (a10, a11, a12), (a20, a21, a22) = a
    det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    return [[(a11 * a22 - a12 * a21) / det, -(a01 * a22 - a02 * a21) / det, (a01 * a12 - a02 * a11) / det],
            [-(a10 * a22 - a12 * a20) / det, (a00 * a22 - a02 * a20) / det, -(a00 * a12 - a02 * a10) / det],
            [(a10 * a21 - a11 * a20) / det, -(a00 * a21 - a01 * a20) / det, (a00 * a11 - a01 * a10) / det]]


def inv4x4kernel(a: arr) -> list:
    """
    Introduction
    ==========
    This function calculates the inverse of a 4x4 matrix.

    Example
    ==========
    >>> inv4x4kernel([[5, 1, 0, 2], [-3, 4, 6, -2], [-9, -3, -6, 7], [-8, -2, -1, -2]])
    [[26.0, 1.0, -2.0, 18.0], [-423.0, -17.0, 32.0, -294.0], [344.0, 14.0, -26.0, 239.0], [147.0, 6.0, -11.0, 102.0]]
    >>>
    :param a: The 4x4 matrix to calculate the inverse of.
    :return: The inverse of the matrix as a 4x4 matrix.
    """
    (a00, a01, a02, a03), (a10, a11, a12, a13), (a20, a21, a22, a23), (a30, a31, a32, a33) = a
    det = det4x4kernel(a)
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    return [[(-a13 * a22 * a31 + a12 * a23 * a31 + a13 * a21 * a32 - a11 * a23 * a32
              - a12 * a21 * a33 + a11 * a22 * a33) / det,
             (a03 * a22 * a31 - a02 * a23 * a31 - a03 * a21 * a32 + a01 * a23 * a32
              + a02 * a21 * a33 - a01 * a22 * a33) / det,
             (-a03 * a12 * a31 + a02 * a13 * a31 + a03 * a11 * a32 - a01 * a13 * a32
              - a02 * a11 * a33 + a01 * a12 * a33) / det,
             (a03 * a12 * a21 - a02 * a13 * a21 - a03 * a11 * a22 + a01 * a13 * a22
              + a02 * a11 * a23 - a01 * a12 * a23) / det],
            [(a13 * a22 * a30 - a12 * a23 * a30 - a13 * a20 * a32 + a10 * a23 * a32
              + a12 * a20 * a33 - a10 * a22 * a33) / det,
             (-a03 * a22 * a30 + a02 * a23 * a30 + a03 * a20 * a32 - a00 * a23 * a32
              - a02 * a20 * a33 + a00 * a22 * a33) / det,
             (a03 * a12 * a30 - a02 * a13 * a30 - a03 * a10 * a32 + a00 * a13 * a32
              + a02 * a10 * a33 - a00 * a12 * a33) / det,
             (-a03 * a12 * a20 + a02 * a13 * a20 + a03 * a10 * a22 - a00 * a13 * a22
              - a02 * a10 * a23 + a00 * a12 * a23) / det],
            [(-a13 * a21 * a30 + a11 * a23 * a30 + a13 * a20 * a31 - a10 * a23 * a31
              - a11 * a20 * a33 + a10 * a21 * a33) / det,
             (a03 * a21 * a30 - a01 * a23 * a30 - a03 * a20 * a31 + a00 * a23 * a31
              + a01 * a20 * a33 - a00 * a21 * a33) / det,
             (-a03 * a11 * a30 + a01 * a13 * a30 + a03 * a10 * a31 - a00 * a13 * a31
              - a01 * a10 * a33 + a00 * a11 * a33) / det,
             (a03 * a11 * a20 - a01 * a13 * a20 - a03 * a10 * a21 + a00 * a13 * a21
              + a01 * a10 * a23 - a00 * a11 * a23) / det],
            [(a12 * a21 * a30 - a11 * a22 * a30 - a12 * a20 * a31 + a10 * a22 * a31
              + a11 * a20 * a32 - a10 * a21 * a32) / det,
             (-a02 * a21 * a30 + a01 * a22 * a30 + a02 * a20 * a31 - a00 * a22 * a31
              - a01 * a20 * a32 + a00 * a21 * a32) / det,
             (a02 * a11 * a30 - a01 * a12 * a30 - a02 * a10 * a31 + a00 * a12 * a31
              + a01 * a10 * a32 - a00 * a11 * a32) / det,
             (-a02 * a11 * a20 + a01 * a12 * a20 + a02 * a10 * a21 - a00 * a12 * a21
              - a01 * a10 * a22 + a00 * a11 * a22) / det]]


def matpow2x2kernel(a: arr, n: num) -> list:
    """
    Introduction
    ==========
    This function raises a 2x2 matrix to a given power n.

    Example
    ==========
    >>> matpow2x2kernel([[1, 2], [2, 4]], 2.722706232293572)
    [[15.999999999999995, 31.99999999999999], [31.99999999999999, 63.99999999999998]]
    >>>
    :param a: The 2x2 matrix to be raised to a power.
    :param n: The power to raise the matrix to.
    :return: The matrix raised to the given power as a 2x2 matrix.
    """
    (a00, a01), (a10, a11) = a
    disc = (a00 ** 2 - 2 * a00 * a11 + 4 * a01 * a10 + a11 ** 2) ** 0.5
    ev_minus = (a00 / 2 + a11 / 2 - disc / 2) ** n
    ev_plus = (a00 / 2 + a11 / 2 + disc / 2) ** n
    denom = a00 ** 2 - 2 * a00 * a11 + a00 * disc + 4 * a01 * a10 + a11 ** 2 - a11 * disc
    return [
        [2 * a01 * a10 * ev_minus / ((a00 - a11 + disc) * disc) - 2 * a01 * a10 * ev_plus / ((a00 - a11 - disc) * disc),
         -4 * a01 ** 2 * a10 * ev_plus / ((a00 - a11 - disc) * denom) - 2 * a01 * ev_minus * (
                 a00 ** 2 - 2 * a00 * a11 + a00 * disc + 2 * a01 * a10 + a11 ** 2 - a11 * disc) / (
                 (a00 - a11 + disc) * denom)],
        [-a10 * ev_minus / disc + a10 * ev_plus / disc,
         2 * a01 * a10 * ev_plus / denom + ev_minus * (
                 a00 ** 2 - 2 * a00 * a11 + a00 * disc + 2 * a01 * a10 + a11 ** 2 - a11 * disc) / denom]
    ]


def matexp2x2kernel(a: arr) -> list:
    """
    Introduction
    ==========
    This function computes the matrix exponential of a 2x2 matrix.

    Example
    ==========
    >>> matexp2x2kernel([[1, 2], [3, 4]])
    [[51.968956198705, 74.73656456700321], [112.10484685050481, 164.0738030492098]]
    >>>
    :param a: The 2x2 matrix to compute the matrix exponential for.
    :return: The matrix exponential of the input matrix as a 2x2 matrix.
    """
    from cmath import exp
    (a00, a01), (a10, a11) = a
    d = (a00 ** 2 - 2 * a00 * a11 + 4 * a01 * a10 + a11 ** 2) ** 0.5
    ed = exp(d)
    if not ed.imag:
        ed = ed.real
    sf = exp((a00 + a11 - d) / 2) / (2 * d)
    if not sf.imag:
        sf = sf.real
    return [[sf * (ed * a00 - a00 - ed * a11 + a11 + ed * d + d), sf * 2 * (ed * a01 - a01)],
            [sf * 2 * (ed * a10 - a10), sf * (-ed * a00 + a00 + ed * a11 - a11 + ed * d + d)]]


def eigen2x2kernel(a: arr) -> tuple:
    """
    Introduction
    ==========
    This function calculates the eigenvalues and eigenvectors of a 2x2 matrix.

    Example
    ==========
    >>> eigen2x2kernel([[1, 2], [3, 4]])
    ([5.372281323269014, -0.3722813232690143], [[0.4574271077563381, 1], [-1.457427107756338, 1]])
    >>>
    :param a: The 2x2 matrix to calculate the eigenvalues and eigenvectors of.
    :return: A tuple containing a list of eigenvalues and a list of corresponding eigenvectors.
    """
    (a00, a01), (a10, a11) = a
    d = (a00 - a11) ** 2 + 4 * a01 * a10
    l1 = 0.5 * (a00 + a11 + d ** 0.5)
    l2 = 0.5 * (a00 + a11 - d ** 0.5)
    v1 = [(a00 - a11 + d ** 0.5) / (2 * a10), 1]
    v2 = [(a00 - a11 - d ** 0.5) / (2 * a10), 1]
    return [l1, l2], [v1, v2]


def lu2x2kernel(a: arr) -> tuple:
    """
    Introduction
    ==========
    This function performs LU decomposition on a 2x2 matrix.

    Example
    ==========
    >>> lu2x2kernel([[1, 2], [3, 4]])
    ([[1, 0], [3.0, 1]], [[1, 2], [0, -2.0]])
    >>>
    :param a: The 2x2 matrix to be decomposed.
    :return: A tuple containing two 2x2 matrices, the lower triangular matrix L and the upper triangular matrix U.
    """
    (a00, a01), (a10, a11) = a
    return [[1, 0], [a10 / a00, 1]], [[a00, a01], [0, a11 - (a01 * a10) / a00]]


def lu3x3kernel(a: arr) -> tuple:
    """
    Introduction
    ==========
    This function performs LU decomposition on a 3x3 matrix.

    Example
    ==========
    >>> lu3x3kernel([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ([[1, 0, 0], [4.0, 1, 0], [7.0, 2.0, 1]], [[1, 2, 3], [0, -3.0, -6.0], [0, 0, 0.0]])
    >>>
    :param a: The 3x3 matrix to be decomposed.
    :return: A tuple containing two 3x3 matrices, the lower triangular matrix L and the upper triangular matrix U.
    """
    (a00, a01, a02), (a10, a11, a12), (a20, a21, a22) = a
    return [[1, 0, 0], [a10 / a00, 1, 0], [a20 / a00, (a21 - a01 * a20 / a00) / (a11 - a01 * a10 / a00), 1]], [
        [a00, a01, a02], [0, a11 - a01 * a10 / a00, a12 - a02 * a10 / a00],
        [0, 0, -a02 * a20 / a00 - (a12 - a02 * a10 / a00) * (a21 - a01 * a20 / a00) / (a11 - a01 * a10 / a00) + a22]]


def lu4x4kernel(a: arr) -> tuple:
    """
    Introduction
    ==========
    This function performs LU decomposition on a 4x4 matrix.

    Example
    ==========
    >>> l, u = lu4x4kernel([[-5, -1, -1, -7], [5, 6, -4, -1], [-5, -6, -6, -6], [-5, -6, 4, -4]])
    >>> l
    [[1, 0, 0, 0], [-1.0, 1, 0, 0], [1.0, -1.0, 1, 0], [1.0, -1.0, -0.0, 1]]
    >>> u
    [[-5, -1, -1, -7], [0, 5.0, -5.0, -8.0], [0, 0, -10.0, -7.0], [0, 0, 0, -5.0]]
    >>>
    :param a: The 4x4 matrix to be decomposed.
    :return: A tuple containing two 4x4 matrices, the lower triangular matrix L and the upper triangular matrix U.
    """
    (a00, a01, a02, a03), (a10, a11, a12, a13), (a20, a21, a22, a23), (a30, a31, a32, a33) = a
    l10 = a10 / a00
    l20 = a20 / a00
    l21 = (a21 - a01 * l20) / (a11 - a01 * l10)
    l30 = a30 / a00
    l31 = (a31 - a01 * l30) / (a11 - a01 * l10)
    u12 = a12 - a02 * l10
    u22 = -a02 * l20 - u12 * l21 + a22
    l32 = (-a02 * l30 - u12 * l31 + a32) / u22
    u11 = a11 - a01 * l10
    u13 = a13 - a03 * l10
    u23 = -a03 * l20 - u13 * l21 + a23
    u33 = -a03 * l30 - u13 * l31 - u23 * l32 + a33
    return [[1, 0, 0, 0], [l10, 1, 0, 0], [l20, l21, 1, 0], [l30, l31, l32, 1]], [
        [a00, a01, a02, a03], [0, u11, u12, u13], [0, 0, u22, u23], [0, 0, 0, u33]]


def matmul8x8kernel(a: arr, b: arr) -> list:
    """
    Introduction
    ==========
    This function performs matrix multiplication for 8x8 matrices.

    Example
    ==========
    >>> matmul8x8kernel([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16],
    ... [17, 18, 19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30, 31, 32],
    ... [33, 34, 35, 36, 37, 38, 39, 40], [41, 42, 43, 44, 45, 46, 47, 48],
    ... [49, 50, 51, 52, 53, 54, 55, 56], [57, 58, 59, 60, 61, 62, 63, 64]],
    ... [[64, 63, 62, 61, 60, 59, 58, 57], [56, 55, 54, 53, 52, 51, 50, 49],
    ... [48, 47, 46, 45, 44, 43, 42, 41], [40, 39, 38, 37, 36, 35, 34, 33],
    ... [32, 31, 30, 29, 28, 27, 26, 25], [24, 23, 22, 21, 20, 19, 18, 17],
    ... [16, 15, 14, 13, 12, 11, 10, 9], [8, 7, 6, 5, 4, 3, 2, 1]])[0]
    [960, 924, 888, 852, 816, 780, 744, 708]
    >>>
    :param a: The first 8x8 matrix to multiply.
    :param b: The second 8x8 matrix to multiply.
    :return: The result of the multiplication as an 8x8 matrix.
    """
    ((a00, a01, a02, a03, a04, a05, a06, a07), (a10, a11, a12, a13, a14, a15, a16, a17),
     (a20, a21, a22, a23, a24, a25, a26, a27), (a30, a31, a32, a33, a34, a35, a36, a37),
     (a40, a41, a42, a43, a44, a45, a46, a47), (a50, a51, a52, a53, a54, a55, a56, a57),
     (a60, a61, a62, a63, a64, a65, a66, a67), (a70, a71, a72, a73, a74, a75, a76, a77)) = a
    ((b00, b01, b02, b03, b04, b05, b06, b07), (b10, b11, b12, b13, b14, b15, b16, b17),
     (b20, b21, b22, b23, b24, b25, b26, b27), (b30, b31, b32, b33, b34, b35, b36, b37),
     (b40, b41, b42, b43, b44, b45, b46, b47), (b50, b51, b52, b53, b54, b55, b56, b57),
     (b60, b61, b62, b63, b64, b65, b66, b67), (b70, b71, b72, b73, b74, b75, b76, b77)) = b
    return [[a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30 + a04 * b40 + a05 * b50 + a06 * b60 + a07 * b70,
             a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31 + a04 * b41 + a05 * b51 + a06 * b61 + a07 * b71,
             a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32 + a04 * b42 + a05 * b52 + a06 * b62 + a07 * b72,
             a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33 + a04 * b43 + a05 * b53 + a06 * b63 + a07 * b73,
             a00 * b04 + a01 * b14 + a02 * b24 + a03 * b34 + a04 * b44 + a05 * b54 + a06 * b64 + a07 * b74,
             a00 * b05 + a01 * b15 + a02 * b25 + a03 * b35 + a04 * b45 + a05 * b55 + a06 * b65 + a07 * b75,
             a00 * b06 + a01 * b16 + a02 * b26 + a03 * b36 + a04 * b46 + a05 * b56 + a06 * b66 + a07 * b76,
             a00 * b07 + a01 * b17 + a02 * b27 + a03 * b37 + a04 * b47 + a05 * b57 + a06 * b67 + a07 * b77],
            [a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30 + a14 * b40 + a15 * b50 + a16 * b60 + a17 * b70,
             a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41 + a15 * b51 + a16 * b61 + a17 * b71,
             a10 * b02 + a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42 + a15 * b52 + a16 * b62 + a17 * b72,
             a10 * b03 + a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43 + a15 * b53 + a16 * b63 + a17 * b73,
             a10 * b04 + a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44 + a15 * b54 + a16 * b64 + a17 * b74,
             a10 * b05 + a11 * b15 + a12 * b25 + a13 * b35 + a14 * b45 + a15 * b55 + a16 * b65 + a17 * b75,
             a10 * b06 + a11 * b16 + a12 * b26 + a13 * b36 + a14 * b46 + a15 * b56 + a16 * b66 + a17 * b76,
             a10 * b07 + a11 * b17 + a12 * b27 + a13 * b37 + a14 * b47 + a15 * b57 + a16 * b67 + a17 * b77],
            [a20 * b00 + a21 * b10 + a22 * b20 + a23 * b30 + a24 * b40 + a25 * b50 + a26 * b60 + a27 * b70,
             a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41 + a25 * b51 + a26 * b61 + a27 * b71,
             a20 * b02 + a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42 + a25 * b52 + a26 * b62 + a27 * b72,
             a20 * b03 + a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43 + a25 * b53 + a26 * b63 + a27 * b73,
             a20 * b04 + a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44 + a25 * b54 + a26 * b64 + a27 * b74,
             a20 * b05 + a21 * b15 + a22 * b25 + a23 * b35 + a24 * b45 + a25 * b55 + a26 * b65 + a27 * b75,
             a20 * b06 + a21 * b16 + a22 * b26 + a23 * b36 + a24 * b46 + a25 * b56 + a26 * b66 + a27 * b76,
             a20 * b07 + a21 * b17 + a22 * b27 + a23 * b37 + a24 * b47 + a25 * b57 + a26 * b67 + a27 * b77],
            [a30 * b00 + a31 * b10 + a32 * b20 + a33 * b30 + a34 * b40 + a35 * b50 + a36 * b60 + a37 * b70,
             a30 * b01 + a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41 + a35 * b51 + a36 * b61 + a37 * b71,
             a30 * b02 + a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42 + a35 * b52 + a36 * b62 + a37 * b72,
             a30 * b03 + a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43 + a35 * b53 + a36 * b63 + a37 * b73,
             a30 * b04 + a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44 + a35 * b54 + a36 * b64 + a37 * b74,
             a30 * b05 + a31 * b15 + a32 * b25 + a33 * b35 + a34 * b45 + a35 * b55 + a36 * b65 + a37 * b75,
             a30 * b06 + a31 * b16 + a32 * b26 + a33 * b36 + a34 * b46 + a35 * b56 + a36 * b66 + a37 * b76,
             a30 * b07 + a31 * b17 + a32 * b27 + a33 * b37 + a34 * b47 + a35 * b57 + a36 * b67 + a37 * b77],
            [a40 * b00 + a41 * b10 + a42 * b20 + a43 * b30 + a44 * b40 + a45 * b50 + a46 * b60 + a47 * b70,
             a40 * b01 + a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41 + a45 * b51 + a46 * b61 + a47 * b71,
             a40 * b02 + a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42 + a45 * b52 + a46 * b62 + a47 * b72,
             a40 * b03 + a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43 + a45 * b53 + a46 * b63 + a47 * b73,
             a40 * b04 + a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44 + a45 * b54 + a46 * b64 + a47 * b74,
             a40 * b05 + a41 * b15 + a42 * b25 + a43 * b35 + a44 * b45 + a45 * b55 + a46 * b65 + a47 * b75,
             a40 * b06 + a41 * b16 + a42 * b26 + a43 * b36 + a44 * b46 + a45 * b56 + a46 * b66 + a47 * b76,
             a40 * b07 + a41 * b17 + a42 * b27 + a43 * b37 + a44 * b47 + a45 * b57 + a46 * b67 + a47 * b77],
            [a50 * b00 + a51 * b10 + a52 * b20 + a53 * b30 + a54 * b40 + a55 * b50 + a56 * b60 + a57 * b70,
             a50 * b01 + a51 * b11 + a52 * b21 + a53 * b31 + a54 * b41 + a55 * b51 + a56 * b61 + a57 * b71,
             a50 * b02 + a51 * b12 + a52 * b22 + a53 * b32 + a54 * b42 + a55 * b52 + a56 * b62 + a57 * b72,
             a50 * b03 + a51 * b13 + a52 * b23 + a53 * b33 + a54 * b43 + a55 * b53 + a56 * b63 + a57 * b73,
             a50 * b04 + a51 * b14 + a52 * b24 + a53 * b34 + a54 * b44 + a55 * b54 + a56 * b64 + a57 * b74,
             a50 * b05 + a51 * b15 + a52 * b25 + a53 * b35 + a54 * b45 + a55 * b55 + a56 * b65 + a57 * b75,
             a50 * b06 + a51 * b16 + a52 * b26 + a53 * b36 + a54 * b46 + a55 * b56 + a56 * b66 + a57 * b76,
             a50 * b07 + a51 * b17 + a52 * b27 + a53 * b37 + a54 * b47 + a55 * b57 + a56 * b67 + a57 * b77],
            [a60 * b00 + a61 * b10 + a62 * b20 + a63 * b30 + a64 * b40 + a65 * b50 + a66 * b60 + a67 * b70,
             a60 * b01 + a61 * b11 + a62 * b21 + a63 * b31 + a64 * b41 + a65 * b51 + a66 * b61 + a67 * b71,
             a60 * b02 + a61 * b12 + a62 * b22 + a63 * b32 + a64 * b42 + a65 * b52 + a66 * b62 + a67 * b72,
             a60 * b03 + a61 * b13 + a62 * b23 + a63 * b33 + a64 * b43 + a65 * b53 + a66 * b63 + a67 * b73,
             a60 * b04 + a61 * b14 + a62 * b24 + a63 * b34 + a64 * b44 + a65 * b54 + a66 * b64 + a67 * b74,
             a60 * b05 + a61 * b15 + a62 * b25 + a63 * b35 + a64 * b45 + a65 * b55 + a66 * b65 + a67 * b75,
             a60 * b06 + a61 * b16 + a62 * b26 + a63 * b36 + a64 * b46 + a65 * b56 + a66 * b66 + a67 * b76,
             a60 * b07 + a61 * b17 + a62 * b27 + a63 * b37 + a64 * b47 + a65 * b57 + a66 * b67 + a67 * b77],
            [a70 * b00 + a71 * b10 + a72 * b20 + a73 * b30 + a74 * b40 + a75 * b50 + a76 * b60 + a77 * b70,
             a70 * b01 + a71 * b11 + a72 * b21 + a73 * b31 + a74 * b41 + a75 * b51 + a76 * b61 + a77 * b71,
             a70 * b02 + a71 * b12 + a72 * b22 + a73 * b32 + a74 * b42 + a75 * b52 + a76 * b62 + a77 * b72,
             a70 * b03 + a71 * b13 + a72 * b23 + a73 * b33 + a74 * b43 + a75 * b53 + a76 * b63 + a77 * b73,
             a70 * b04 + a71 * b14 + a72 * b24 + a73 * b34 + a74 * b44 + a75 * b54 + a76 * b64 + a77 * b74,
             a70 * b05 + a71 * b15 + a72 * b25 + a73 * b35 + a74 * b45 + a75 * b55 + a76 * b65 + a77 * b75,
             a70 * b06 + a71 * b16 + a72 * b26 + a73 * b36 + a74 * b46 + a75 * b56 + a76 * b66 + a77 * b76,
             a70 * b07 + a71 * b17 + a72 * b27 + a73 * b37 + a74 * b47 + a75 * b57 + a76 * b67 + a77 * b77]]
