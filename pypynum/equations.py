def lin_eq(left: list, right: list) -> list:
    from .Matrix import mat
    d = mat(left).det()
    if d != 0:
        return [mat([left[_][:item] + [right[_]] + left[_][item + 1:] for _ in range(len(left))]).det() / d
                for item in range(len(left))]
    return [float("inf")] * len(right)


def poly_eq(coefficients: list) -> list:
    from .Matrix import eigen, mat
    p = [_ / coefficients[0] for _ in coefficients[1:]]
    return sorted(eigen(mat([[-p[i] if j == 0 else 1 if i + 1 == j else 0 for j in range(len(p))]
                             for i in range(len(p))]))[0].diag().t()[0], key=lambda c: (c.real, c.imag))
