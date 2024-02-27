def linear_equation(left: list, right: list) -> list:
    from .Array import array
    from .Matrix import mat
    d = mat(left).det()
    if d != 0:
        return array([mat([left[_][:item] + [right[_]] + left[_][item + 1:] for _ in range(len(left))]).det() / d
                      for item in range(len(left))])
    return array([float("inf")] * len(right))


def polynomial_equation(coefficients: list) -> list:
    from .Array import array
    from .Matrix import mat
    from .__temporary import eigenvalue as eig
    p = [_ / coefficients[0] for _ in coefficients[1:]]
    return array(eig(mat([[-p[i] if j == 0 else 1 if i + 1 == j else 0
                           for j in range(len(p))] for i in range(len(p))])))
