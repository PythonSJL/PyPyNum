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
