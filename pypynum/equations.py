from .Matrix import mat


def multivariate_linear_equation_system(left: list, right: list) -> list:
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
    d = mat(left).det()
    if d == 0:
        raise ValueError("determinant == 0")
    for item in range(len(left)):
        result.append(mat([left[_][:item] + [right[_]] + left[_][item + 1:] for _ in range(len(left))]).det() / d)
    return [round(_, 12) for _ in result]


mles = multivariate_linear_equation_system
