import Matrix


def solve_equations(a1, b1, c1, a2, b2, c2):
    """
    solve two linear equations of two variables
    ax + by = c
    dx + ey = f
    return (x, y) if the equations are consistent
    return None otherwise
    """
    # calculate denominator
    denominator = a1 * b2 - a2 * b1
    if denominator == 0:
        return None
    # calculate x and y
    x = (c1 * b2 - c2 * b1) / denominator
    y = (a1 * c2 - a2 * c1) / denominator
    return x, y


def solve_equations_pro(left: list, right: list) -> list:
    # # # check # # #
    if type(left) == list:
        num = len(left)
    else:
        raise ValueError("type(left) != list")
    for _ in range(num):
        if type(left[_]) != list:
            raise ValueError("type(left[{}]) != list".format(_))
        elif len(left[_]) != num:
            raise ValueError("len(left[{}]) != len(left)".format(_))
    if type(right) == list:
        if len(right) != num:
            raise ValueError("len(right) != len(left)")
    else:
        raise ValueError("type(right) != list")
    # # # start # # #
    result = []
    m = Matrix.Matrix(left)
    d = m.det()
    if d == 0:
        raise ValueError("determinant == 0")
    for item in range(len(left)):
        M = Matrix.Matrix([left[_][:item] + [right[_]] + left[_][item + 1:] for _ in range(len(left))])
        result.append(M.det() / d)
    return [round(_, 12) for _ in result]
