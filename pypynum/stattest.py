def mediantest(*samples, ties="below", lambda_=1, corr=True):
    from math import isnan
    from .Matrix import mat
    from .maths import median
    if len(samples) < 2:
        raise ValueError("median_test requires two or more samples.")
    data = [list(sample) for sample in samples]
    cdata = [item for sublist in data for item in sublist]
    grand_median = median([x for x in cdata if x is not None and not (isinstance(x, float) and isnan(x))])
    table = [[0] * len(data) for _ in range(2)]
    for k, sample in enumerate(data):
        above = sum([x > grand_median for x in sample if x is not None
                     and not (isinstance(x, float) and isnan(x))])
        below = sum([x < grand_median for x in sample if x is not None
                     and not (isinstance(x, float) and isnan(x))])
        equal = len(sample) - (above + below)
        table[0][k] += above
        table[1][k] += below
        if ties == "below":
            table[1][k] += equal
        elif ties == "above":
            table[0][k] += equal
        elif ties != "ignore":
            raise ValueError("Invalid ties option: {}. Valid options are: ['below', 'above', 'ignore']".format(ties))
    stat, p, _, _ = chi2_cont(table, lambda_=lambda_, corr=corr)
    return stat, p, grand_median, mat(table)


def skewtest(data: list, two_tailed: bool = True) -> tuple:
    from math import log, sqrt
    from .dists import normal_cdf
    from .maths import skew
    n = len(data)
    y = skew(data) * sqrt(((n + 1) * (n + 3)) / (6 * (n - 2)))
    beta = 3 * (n ** 2 + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
    w = sqrt(2 * (beta - 1)) - 1
    delta = 1 / sqrt(0.5 * log(w))
    alpha = sqrt(2 / (w - 1))
    z = delta * log(y / alpha + sqrt((y / alpha) ** 2 + 1))
    p_value = normal_cdf(z) if z > 0 else 1 - normal_cdf(z)
    if two_tailed:
        p_value = 2 * min(p_value, 1 - p_value)
    return z, p_value


def kurttest(data: list, two_tailed: bool = True) -> tuple:
    from math import sqrt
    from .dists import normal_cdf
    from .maths import kurt, sign
    n = len(data)
    b = kurt(data, fisher=False)
    e = 3 * (n - 1) / (n + 1)
    vb = 24 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1) * (n + 3) * (n + 5))
    x = (b - e) / sqrt(vb)
    sb = 6 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) * sqrt((6 * (n + 3) * (n + 5)) / (n * (n - 2) * (n - 3)))
    a = 6 + 8 / sb * (2 / sb + sqrt(1 + 4 / (sb ** 2)))
    d = 1 + x * sqrt(2 / (a - 4))
    t = sign(d) * (d == 0 and float("nan") or ((1 - 2 / a) / abs(d)) ** (1 / 3))
    z = (1 - 2 / (9 * a) - t) / sqrt(2 / (9 * a))
    p_value = normal_cdf(z) if z > 0 else 1 - normal_cdf(z)
    if two_tailed:
        p_value = 2 * min(p_value, 1 - p_value)
    return z, p_value


def normaltest(data: list) -> tuple:
    from .dists import chi2_cdf
    s = skewtest(data)[0]
    k = kurttest(data)[0]
    n = s * s + k * k
    return n, 1 - chi2_cdf(2, n)


def chisquare(observed: list, expected: list = None) -> tuple:
    from .dists import chi2_pdf
    from .maths import integ
    if expected is None:
        expected = [1 / len(observed)] * len(observed)
    chi2_stat = sum([(o - e) ** 2 / e for o, e in zip(observed, expected)])
    dof = len(observed) - 1
    try:
        p_value = integ(chi2_pdf, chi2_stat, 1000, df=dof)
    except OverflowError:
        p_value = float("nan")
    return chi2_stat, p_value


def chi2_cont(contingency: list, lambda_: float = 1.0, calc_p: bool = True, corr: bool = True) -> tuple:
    from .Matrix import mat
    from .dists import chi2_cdf
    from .maths import xlogy
    contingency = mat(contingency)
    dof = (contingency.rows - 1) * (contingency.cols - 1)
    row_sums = [sum(row) for row in contingency]
    col_sums = [sum([contingency[i][j] for i in range(contingency.rows)]) for j in range(contingency.cols)]
    grand_total = sum(row_sums)
    expected = [[row_sums[i] * col_sums[j] / grand_total for j in range(contingency.cols)]
                for i in range(contingency.rows)]
    if dof == 1 and corr:
        corrected = [[0, 0], [0, 0]]
        for i in range(2):
            for j in range(2):
                diff = expected[i][j] - contingency[i][j]
                direction = 1 if diff > 0 else -1 if diff < 0 else 0
                magnitude = min(0.5, abs(diff))
                corrected[i][j] = contingency[i][j] + direction * magnitude
        contingency = mat(corrected)
    chi2 = 0
    for i in range(contingency.rows):
        for j in range(contingency.cols):
            observed_freq = contingency[i][j]
            expected_freq = expected[i][j]
            if expected_freq > 0:
                if lambda_ == 1:
                    terms = (observed_freq - expected_freq) ** 2 / expected_freq
                elif lambda_ == 0:
                    terms = 2.0 * xlogy(observed_freq, observed_freq / expected_freq)
                elif lambda_ == -1:
                    terms = 2.0 * xlogy(expected_freq, expected_freq / observed_freq)
                else:
                    terms = observed_freq * ((observed_freq / expected_freq) ** lambda_ - 1)
                    terms /= 0.5 * lambda_ * (lambda_ + 1)
                chi2 += terms
    try:
        p = 1 - chi2_cdf(chi2, df=dof) if calc_p else None
    except OverflowError:
        p = float("nan")
    return chi2, p, dof, mat(expected)
