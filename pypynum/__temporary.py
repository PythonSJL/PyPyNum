def eigenvalue(matrix, eps=1e-100):
    from math import hypot
    from cmath import sqrt
    from .Matrix import mat
    A = matrix.copy()
    n = A.rows
    if n == 1:
        return [A[0]], mat([[1]])
    T = [0j] * n
    if n > 2:
        for i in range(n - 1, 1, -1):
            scale = 0
            for k in range(0, i):
                scale += abs(A[i, k].real) + abs(A[i, k].imag)
            scale_inv = 0
            if scale != 0:
                scale_inv = 1 / scale
            if scale == 0 or scale_inv == float("inf"):
                T[i] = 0
                A[i, i - 1] = 0
                continue
            H = 0
            for k in range(0, i):
                A[i, k] = A[i, k] * scale_inv
                rr, ii = A[i, k].real, A[i, k].imag
                H += rr * rr + ii * ii
            F = A[i, i - 1]
            f = abs(F)
            G = sqrt(H)
            A[i, i - 1] = - G * scale
            if f == 0:
                T[i] = G
            else:
                ff = F / f
                T[i] = F + G * ff
                A[i, i - 1] = A[i, i - 1] * ff
            H += G * f
            H = 1 / sqrt(H)
            T[i] *= H
            for k in range(0, i - 1):
                A[i, k] = A[i, k] * H
            for j in range(0, i):
                G = T[i].conjugate() * A[j, i - 1]
                for k in range(0, i - 1):
                    G += A[i, k].conjugate() * A[j, k]
                A[j, i - 1] = A[j, i - 1] - G * T[i]
                for k in range(0, i - 1):
                    A[j, k] = A[j, k] - G * A[i, k]
            for j in range(0, n):
                G = T[i] * A[i - 1, j]
                for k in range(0, i - 1):
                    G += A[i, k] * A[k, j]
                A[i - 1, j] = A[i - 1, j] - G * T[i].conjugate()
                for k in range(0, i - 1):
                    A[k, j] = A[k, j] - G * A[i, k].conjugate()
    Q = A.copy()
    for x in range(n):
        for y in range(x + 2, n):
            Q[y, x] = 0
    norm = 0
    for x in range(n):
        for y in range(min(x + 2, n)):
            norm += abs(Q[y, x])
    norm = norm ** 0.5 / n
    if norm == 0:
        return
    n0 = 0
    n1 = n
    eps0 = eps / (100 * n)
    its = 0
    while True:
        k = n0
        while k + 1 < n1:
            s = abs(Q[k, k].real) + abs(Q[k, k].imag) + abs(Q[k + 1, k + 1].real) + abs(Q[k + 1, k + 1].imag)
            if s < eps0 * norm:
                s = norm
            if abs(Q[k + 1, k]) < eps0 * s:
                break
            k += 1
        if k + 1 < n1:
            Q[k + 1, k] = 0
            n0 = k + 1
            its = 0
            if n0 + 1 >= n1:
                n0 = 0
                n1 = k + 1
                if n1 < 2:
                    break
        else:
            if its % 30 == 10:
                shift = Q[n1 - 1, n1 - 2]
            elif its % 30 == 20:
                shift = abs(Q[n1 - 1, n1 - 2])
            elif its % 30 == 29:
                shift = norm
            else:
                t = Q[n1 - 2, n1 - 2] + Q[n1 - 1, n1 - 1]
                s = (Q[n1 - 1, n1 - 1] - Q[n1 - 2, n1 - 2]) ** 2 + 4 * Q[n1 - 1, n1 - 2] * Q[n1 - 2, n1 - 1]
                if s.real > 0:
                    s = sqrt(s)
                else:
                    s = sqrt(-s) * 1j
                a = (t + s) / 2
                b = (t - s) / 2
                if abs(Q[n1 - 1, n1 - 1] - a) > abs(Q[n1 - 1, n1 - 1] - b):
                    shift = b
                else:
                    shift = a
            its += 1
            c = Q[n0, n0] - shift
            s = Q[n0 + 1, n0]
            v = hypot(abs(c), abs(s))
            if v == 0:
                c = 1
                s = 0
            else:
                c /= v
                s /= v
            cc = c.conjugate()
            cs = s.conjugate()
            for k in range(n0, n):
                x = Q[n0, k]
                y = Q[n0 + 1, k]
                Q[n0, k] = cc * x + cs * y
                Q[n0 + 1, k] = c * y - s * x
            for k in range(min(n1, n0 + 3)):
                x = Q[k, n0]
                y = Q[k, n0 + 1]
                Q[k, n0] = c * x + s * y
                Q[k, n0 + 1] = cc * y - cs * x
            if not isinstance(A, bool):
                for k in range(n):
                    x = A[k, n0]
                    y = A[k, n0 + 1]
                    A[k, n0] = c * x + s * y
                    A[k, n0 + 1] = cc * y - cs * x
            for j in range(n0, n1 - 2):
                c = Q[j + 1, j]
                s = Q[j + 2, j]
                v = hypot(abs(c), abs(s))
                if v == 0:
                    Q[j + 1, j] = 0
                    c = 1
                    s = 0
                else:
                    Q[j + 1, j] = v
                    c /= v
                    s /= v
                Q[j + 2, j] = 0
                cc = c.conjugate()
                cs = s.conjugate()
                for k in range(j + 1, n):
                    x = Q[j + 1, k]
                    y = Q[j + 2, k]
                    Q[j + 1, k] = cc * x + cs * y
                    Q[j + 2, k] = c * y - s * x
                for k in range(0, min(n1, j + 4)):
                    x = Q[k, j + 1]
                    y = Q[k, j + 2]
                    Q[k, j + 1] = c * x + s * y
                    Q[k, j + 2] = cc * y - cs * x
                if not isinstance(A, bool):
                    for k in range(0, n):
                        x = A[k, j + 1]
                        y = A[k, j + 2]
                        A[k, j + 1] = c * x + s * y
                        A[k, j + 2] = cc * y - cs * x
    E = [Q[_, _] for _ in range(n)]
    return E
