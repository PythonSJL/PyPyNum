from math import (atan as _atan, cos as _cos, erf as _erf, exp as _exp,
                  factorial as _factorial, gamma as _gamma, log as _log, sqrt as _sqrt)


def beta_pdf(x, a=1.0, b=1.0):
    """
    Introduction
    ==========
    Beta probability density function

    Example
    ==========
    >>> beta_pdf(0.5, 2, 2)
    1.5
    >>>
    :param x: The value at which to evaluate the PDF, must be between 0 and 1.
    :param a: The first shape parameter, must be positive.
    :param b: The second shape parameter, must be positive.
    :return: The probability density at x for the Beta distribution with parameters a and b.
    """
    if a <= 0 or b <= 0:
        raise ValueError("Shape parameters a and b must be positive")
    if x < 0 or x > 1:
        return 0
    return (x ** (a - 1) * (1 - x) ** (b - 1)) / _gamma(a) / _gamma(b) * _gamma(a + b)


def binom_pmf(k, n, p):
    """
    Introduction
    ==========
    Binomial probability mass function

    Example
    ==========
    >>> binom_pmf(2, 4, 0.5)
    0.375
    >>>
    :param k: The number of successes, must be an integer between 0 and n.
    :param n: The number of trials, must be a positive integer.
    :param p: The probability of success in a single trial, must be between 0 and 1.
    :return: The probability of k successes in n trials with probability p of success in each trial.
    """
    if not 0 <= p <= 1:
        raise ValueError("Probability p must be between 0 and 1")
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    if not isinstance(n, int) or not isinstance(k, int):
        raise ValueError("n and k must be integers")
    return _factorial(n) / (_factorial(k) * _factorial(n - k)) * (p ** k) * ((1 - p) ** (n - k))


def cauchy_pdf(x, x0, gamma):
    """
    Introduction
    ==========
    Cauchy probability density function

    Example
    =========
    >>> cauchy_pdf(0, 0, 1)
    0.3183098861837907
    >>>
    :param x: The value at which to evaluate the PDF.
    :param x0: The location parameter of the distribution.
    :param gamma: The scale parameter of the distribution, must be positive.
    :return: The probability density at x for the Cauchy distribution with location parameter x0
        and scale parameter gamma.
    """
    if gamma <= 0:
        raise ValueError("Scale parameter gamma must be positive")
    return 1 / (3.141592653589793 * gamma * (1 + ((x - x0) / gamma) ** 2))


def cauchy_cdf(x, x0, gamma):
    """
    Cauchy cumulative distribution function

    Example
    =========
    >>> cauchy_cdf(0, 0, 1)
    0.5
    >>>
    :param x: The value at which to evaluate the CDF.
    :param x0: The location parameter of the distribution.
    :param gamma: The scale parameter of the distribution, must be positive.
    :return: The cumulative probability at x for the Cauchy distribution with location parameter x0
        and scale parameter gamma.
    """
    if gamma <= 0:
        raise ValueError("Scale parameter gamma must be positive")
    return 0.5 + _atan((x - x0) / gamma) / 3.141592653589793


def chi2_pdf(x, df=1):
    """
    Introduction
    ==========
    Chi-squared probability density function

    Example
    ==========
    >>> chi2_pdf(2, 2)
    0.18393972058572117
    >>>
    :param x: The value at which to evaluate the PDF, must be non-negative.
    :param df: The degrees of freedom, must be positive.
    :return: The probability density at x for the Chi-squared distribution with df degrees of freedom.
    """
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")
    if x < 0:
        return 0
    try:
        return (x ** (df / 2 - 1) * _exp(-x / 2)) / (2 ** (df / 2) * _gamma(df / 2))
    except ZeroDivisionError:
        return 0


def chi2_cdf(x, df=1):
    """
    Introduction
    ==========
    Chi-squared cumulative distribution function

    Example
    ==========
    >>> chi2_cdf(2, 2)
    0.6321205588285578
    >>>
    :param x: The value at which to evaluate the CDF.
    :param df: The degrees of freedom, must be positive.
    :return: The probability density at x for the Chi-squared distribution with df degrees of freedom.
    """
    from math import gamma
    from .maths import lower_gamma
    return lower_gamma(df / 2, x / 2) / gamma(df / 2)


def expon_pdf(x, scale=1.0):
    """
    Introduction
    ==========
    Exponential probability density function

    Example
    ==========
    >>> expon_pdf(1, 1)
    0.36787944117144233
    >>>
    :param x: The value at which to evaluate the PDF, must be non-negative.
    :param scale: The scale parameter, must be positive.
    :return: The probability density at x for the Exponential distribution with the given scale parameter.
    """
    if scale <= 0:
        raise ValueError("Scale parameter must be positive")
    return 1 / scale * _exp(-x / scale) if x >= 0 else 0


def expon_cdf(x, scale=1.0):
    """
    Introduction
    ==========
    Exponential cumulative distribution function

    Example
    ==========
    >>> expon_cdf(1, 1)
    0.6321205588285577
    >>>
    :param x: The value at which to evaluate the CDF, must be non-negative.
    :param scale: The scale parameter, must be positive.
    :return: The cumulative probability at x for the Exponential distribution with the given scale parameter.
    """
    if scale <= 0:
        raise ValueError("Scale parameter must be positive")
    if x < 0:
        return 0
    return 1 - _exp(-x / scale)


def f_pdf(x, dfnum=1, dfden=1):
    """
    Introduction
    ==========
    F probability density function

    Example
    ==========
    >>> f_pdf(1.5, 2, 3)
    0.17677669529663687
    >>>
    :param x: The value at which to evaluate the PDF, must be non-negative.
    :param dfnum: The degrees of freedom of the numerator, must be positive.
    :param dfden: The degrees of freedom of the denominator, must be positive.
    :return: The probability density at x for the F distribution with dfnum and dfden degrees of freedom.
    """
    if dfnum <= 0 or dfden <= 0:
        raise ValueError("Degrees of freedom parameters must be positive")
    if x <= 0:
        return 0
    return (_gamma((dfnum + dfden) / 2) / (_gamma(dfnum / 2) * _gamma(dfden / 2))) * (dfnum / dfden) ** (
            dfnum / 2) * x ** (dfnum / 2 - 1) / (1 + dfnum * x / dfden) ** ((dfnum + dfden) / 2)


def gamma_pdf(x, shape=1.0, scale=1.0):
    """
    Introduction
    ==========
    Gamma probability density function

    Example
    ==========
    >>> gamma_pdf(2, 1, 1)
    0.1353352832366127
    >>>
    :param x: The value at which to evaluate the PDF, must be non-negative.
    :param shape: The shape parameter, must be positive.
    :param scale: The scale parameter, must be positive.
    :return: The probability density at x for the Gamma distribution with the given shape and scale parameters.
    """
    if shape <= 0 or scale <= 0:
        raise ValueError("Shape and scale parameters must be positive")
    if x < 0:
        return 0
    return (x ** (shape - 1) * _exp(-x / scale)) / (scale ** shape * _gamma(shape))


def geometric_pmf(k, p=0.5):
    """
    Introduction
    ==========
    Geometric probability mass function

    Example
    ==========
    >>> geometric_pmf(2, 0.5)
    0.25
    >>>
    :param k: The number of trials until the first success, must be a non-negative integer.
    :param p: The probability of success on each trial, must be between 0 and 1.
    :return: The probability of getting the first success on the k-th trial.
    """
    if not 0 <= p <= 1:
        raise ValueError("Probability p must be between 0 and 1")
    if k < 0:
        raise ValueError("k must be non-negative")
    return p * (1 - p) ** (k - 1)


def hypergeom_pmf(k, mg, n, nt):
    """
    Introduction
    ==========
    Hypergeometric probability mass function

    Example
    ==========
    >>> hypergeom_pmf(2, 5, 3, 2)
    0.3
    >>>
    :param k: The number of successes, must be a non-negative integer.
    :param mg: The total number of items in the population, must be a non-negative integer.
    :param n: The number of items in the sample, must be a non-negative integer less than or equal to mg.
    :param nt: The number of successes in the population, must be a non-negative integer less than or equal to mg.
    :return: The probability of drawing k successes in a sample of size n from a population
        with mg items and nt successes.
    """
    try:
        from math import comb
    except ImportError:
        from .maths import combination as comb
    if not (isinstance(k, int) and k >= 0) or not (isinstance(mg, int) and mg >= 0) or not (
            isinstance(n, int) and n >= 0) or not (isinstance(nt, int) and nt >= 0):
        raise ValueError("All parameters must be non-negative integers")
    if n > mg:
        raise ValueError("n must be less than or equal to mg")
    if k > n:
        raise ValueError("k must be less than or equal to n")
    if nt > mg:
        raise ValueError("nt must be less than or equal to mg")
    if k > nt:
        raise ValueError("k must be less than or equal to nt")
    return comb(nt, k) * comb(mg - nt, n - k) / comb(mg, n)


def inv_gauss_pdf(x, mu, lambda_, alpha):
    """
    Introduction
    ==========
    Inverse Gaussian (inverse normal) probability density function

    Example
    =========
    >>> inv_gauss_pdf(1, 1, 1, 1)
    0.3989422804014327
    >>>
    :param x: The value at which to evaluate the PDF. It must be positive.
    :param mu: The mean of the distribution, must be positive.
    :param lambda_: The shape parameter of the distribution, must be positive.
    :param alpha: The scale parameter of the distribution, must be non-negative.
    :return: The probability density at x for the inverse Gaussian distribution
        with mean mu, shape lambda_, and scale alpha.
    """
    if mu <= 0 or lambda_ <= 0 or alpha < 0:
        raise ValueError("mu, lambda_, and alpha parameters must be positive, and alpha must be non-negative")
    if x <= 0:
        raise ValueError("x must be positive")
    return _sqrt(lambda_ / (6.283185307179586 * x ** 3)) / alpha * _exp(-lambda_ * (x - mu) ** 2 / (
            2 * mu ** 2 * x)) * (1 + alpha * (x - mu) / mu ** 2) ** (-0.5)


def levy_pdf(x, c):
    """
    Introduction
    ==========
    Lévy probability density function

    Example
    =========
    >>> levy_pdf(1, 1)
    0.24197072451914337
    >>>
    :param x: The value at which to evaluate the PDF. It must be positive.
    :param c: The scale parameter of the distribution, must be positive.
    :return: The probability density at x for the Lévy distribution with scale parameter c.
    """
    if c <= 0:
        raise ValueError("Scale parameter c must be positive")
    if x <= 0:
        raise ValueError("x must be positive")
    return _sqrt(c / 6.283185307179586) * _exp(-c / (2 * x)) / (x ** 1.5)


def log_logistic_pdf(x, alpha, beta):
    """
    Introduction
    ==========
    Log-logistic probability density function

    Example
    =========
    >>> log_logistic_pdf(1, 1, 1)
    0.25
    >>>
    :param x: The value at which to evaluate the PDF. It must be positive.
    :param alpha: The scale parameter of the distribution, must be positive.
    :param beta: The shape parameter of the distribution, must be positive.
    :return: The probability density at x for the log-logistic distribution with scale parameter alpha
        and shape parameter beta.
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta parameters must be positive")
    if x <= 0:
        raise ValueError("x must be positive")
    return (beta / alpha) * (x / alpha) ** (beta - 1) * (1 + (x / alpha) ** beta) ** (-2)


def log_logistic_cdf(x, alpha, beta):
    """
    Introduction
    ==========
    Log-logistic cumulative distribution function

    Example
    =========
    >>> log_logistic_cdf(1, 1, 1)
    0.5906161091496412
    >>>
    :param x: The value at which to evaluate the CDF. It must be positive.
    :param alpha: The scale parameter of the distribution, must be positive.
    :param beta: The shape parameter of the distribution, must be positive.
    :return: The cumulative probability at x for the log-logistic distribution with scale parameter alpha
        and shape parameter beta.
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta parameters must be positive")
    if x <= 0:
        raise ValueError("x must be positive")
    return alpha / (alpha + beta * _log(1 + (x / beta) ** alpha))


def logistic_pdf(x, mu=0, s=1):
    """
    Introduction
    ==========
    Logistic probability density function

    Example
    =========
    >>> logistic_pdf(0, 0, 1)
    0.25
    >>>
    :param x: The value at which to evaluate the PDF.
    :param mu: The location parameter of the distribution.
    :param s: The scale parameter of the distribution, must be positive.
    :return: The probability density at x for the logistic distribution with location parameter mu
        and scale parameter s.
    """
    if s <= 0:
        raise ValueError("Scale parameter s must be positive")
    return _exp(-(x - mu) / s) / (s * (1 + _exp(-(x - mu) / s)) ** 2)


def logistic_cdf(x, mu=0, s=1):
    """
    Introduction
    ==========
    Logistic cumulative distribution function

    Example
    =========
    >>> logistic_cdf(0, 0, 1)
    0.5
    >>>
    :param x: The value at which to evaluate the CDF.
    :param mu: The location parameter of the distribution.
    :param s: The scale parameter of the distribution, must be positive.
    :return: The cumulative probability at x for the logistic distribution with location parameter mu
        and scale parameter s.
    """
    if s <= 0:
        raise ValueError("Scale parameter s must be positive")
    return 1 / (1 + _exp(-(x - mu) / s))


def lognorm_pdf(x, s=1, scale=1):
    """
    Introduction
    ==========
    Lognormal probability density function

    Example
    ==========
    >>> lognorm_pdf(2, 1, 1)
    0.1568740192789811
    >>>
    :param x: The value at which to evaluate the PDF, must be non-negative.
    :param s: The shape parameter, must be positive.
    :param scale: The scale parameter, must be positive.
    :return: The probability density at x for the Lognormal distribution with the given shape and scale parameters.
    """
    if s <= 0 or scale <= 0:
        raise ValueError("Shape and scale parameters must be positive")
    if x <= 0:
        return 0
    return 1 / (x * s * 2.5066282746310002) * _exp(-((_log(x) - _log(scale)) ** 2) / (2 * s ** 2))


def lognorm_cdf(x, mu, sigma):
    """
    Introduction
    ==========
    Lognormal cumulative distribution function

    Example
    ==========
    >>> lognorm_cdf(1, 0, 1)
    0.5
    >>>
    :param x: The value at which to evaluate the CDF, must be non-negative.
    :param mu: The mean of the logarithm of the distribution.
    :param sigma: The standard deviation of the logarithm of the distribution.
    :return: The cumulative probability at x for the Lognormal distribution with the given mu and sigma.
    """
    if x <= 0:
        return 0
    z = (_log(x) - mu) / sigma
    return normal_cdf(z)


def logser_pmf(k, p):
    """
    Introduction
    ==========
    Logarithmic (log-series) probability mass function

    Example
    ==========
    >>> logser_pmf(2, 0.5)
    0.045084220027780106
    >>>
    :param k: The number of failures before the first success, must be a non-negative integer.
    :param p: The probability of success on each trial, must be a positive number less than 1.
    :return: The probability of observing k failures before the first success in a series of Bernoulli trials
        with success probability p.
    """
    if not (isinstance(k, int) and k >= 0):
        raise ValueError("k must be a non-negative integer")
    if not (isinstance(p, (int, float)) and 0 < p < 1):
        raise ValueError("p must be a positive number less than 1")
    if k == 0:
        return p
    else:
        return -p * (1 - p) ** (k + 1) / (k * _log(1 - p))


def multinomial_pmf(k, n, p):
    """
    Introduction
    ==========
    Multinomial probability mass function

    Example
    =========
    >>> multinomial_pmf([2, 1, 0], 3, [0.5, 0.4, 0.1])
    0.30000000000000004
    >>>
    :param k: A list of integers representing the number of successes in each category.
        The length of k must match the length of p.
    :param n: The total number of trials.
    :param p: A list of probabilities for each category. The sum of p must be equal to 1.
    :return: The probability mass for the given number of successes in each category.
    """
    if len(k) != len(p):
        raise ValueError("The length of k and p must be the same")
    if sum(k) != n:
        raise ValueError("The sum of k values must be equal to n")
    p_sum = sum(p)
    p_normalized = [pi / p_sum for pi in p]
    prob_product = 1
    for ki, pi in zip(k, p_normalized):
        prob_product *= pi ** ki
    denominator = 1
    for ki in k:
        denominator *= _factorial(ki)
    return _factorial(n) / denominator * prob_product


def nbinom_pmf(k, n=1, p=0.5):
    """
    Introduction
    ==========
    Negative binomial probability mass function

    Example
    ==========
    >>> nbinom_pmf(1, 1, 0.5)
    0.25
    >>>
    :param k: The number of successes, must be a non-negative integer.
    :param n: The number of failures before the experiment is stopped, must be a positive integer.
    :param p: The probability of success on each trial, must be between 0 and 1.
    :return: The probability of getting k successes before n failures occur in a sequence of trials
        with probability p of success on each trial.
    """
    if not 0 <= p <= 1:
        raise ValueError("Probability p must be between 0 and 1")
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    if not isinstance(n, int) or not isinstance(k, int):
        raise ValueError("n and k must be integers")
    return _factorial(n + k - 1) / (_factorial(k) * _factorial(n)) * (p ** n) * ((1 - p) ** k)


def nhypergeom_pmf(k, m, n, r):
    """
    Introduction
    ==========
    Negative hypergeometric probability mass function

    Example
    ==========
    >>> nhypergeom_pmf(2, 10, 4, 2)
    0.3333333333333333
    >>>
    :param k: The number of failures before getting r successes, must be a non-negative integer.
    :param m: The total number of items in the population, must be a non-negative integer.
    :param n: The total sample size, must be a non-negative integer less than or equal to M.
    :param r: The number of successes, must be a non-negative integer less than or equal to n.
    :return: The probability of observing k failures before observing r successes in a sample of size n
        from a population with M items.
    """
    try:
        from math import comb
    except ImportError:
        from .maths import combination as comb
    if not (isinstance(k, int) and k >= 0) or not (isinstance(m, int) and m >= 0) or not (
            isinstance(n, int) and n >= 0) or not (isinstance(r, int) and r >= 0):
        raise ValueError("All parameters must be non-negative integers")
    if n > m:
        raise ValueError("n must be less than or equal to m")
    if k + r > n:
        raise ValueError("k + r must be less than or equal to n")
    if r > m:
        raise ValueError("r must be less than or equal to m")
    return comb(n - k - 1, r - 1) * comb(m - n, k) / comb(m, r)


def normal_pdf(x, mu=0, sigma=1):
    """
    Introduction
    ==========
    Normal (Gaussian) probability density function

    Example
    ==========
    >>> normal_pdf(0, 0, 1)
    0.3989422804014327
    >>>
    :param x: The value at which to evaluate the PDF.
    :param mu: The mean or location parameter of the distribution.
    :param sigma: The standard deviation or scale parameter of the distribution, must be positive.
    :return: The probability density at x for the normal distribution with mean mu and standard deviation sigma.
    """
    if sigma <= 0:
        raise ValueError("Sigma parameter must be positive")
    return (1 / (sigma * 2.5066282746310002)) * _exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def normal_cdf(x, mu=0, sigma=1):
    """
    Introduction
    ==========
    Normal (Gaussian) cumulative distribution function

    Example
    ==========
    >>> normal_cdf(0, 0, 1)
    0.5
    >>>
    :param x: The value at which to evaluate the CDF.
    :param mu: The mean or location parameter of the distribution.
    :param sigma: The standard deviation or scale parameter of the distribution, must be positive.
    :return: The cumulative probability at x for the normal distribution with mean mu and standard deviation sigma.
    """
    if sigma <= 0:
        raise ValueError("Sigma parameter must be positive")
    return 0.5 * (1 + _erf((x - mu) / (sigma * _sqrt(2))))


def pareto_pdf(x, k, m):
    """
    Introduction
    ==========
    Pareto probability density function

    Example
    =========
    >>> pareto_pdf(10, 2, 5)
    0.00040000000000000013
    >>>
    :param x: The value at which to evaluate the PDF. It must be greater than or equal to the scale parameter m.
    :param k: The shape parameter of the distribution, must be positive.
    :param m: The scale parameter of the distribution, must be positive.
    :return: The probability density at x for the Pareto distribution with shape parameter k and scale parameter m.
    """
    if k <= 0 or m <= 0:
        raise ValueError("Both k and m parameters must be positive")
    if x < m:
        raise ValueError("x must be greater than or equal to the scale parameter m")
    return k / m * (1 / x) ** (k + 1)


def poisson_pmf(k, mu=1):
    """
    Introduction
    ==========
    Poisson probability mass function

    Example
    ==========
    >>> poisson_pmf(1, 1)
    0.36787944117144233
    >>>
    :param k: The number of successes (events occurred), must be non-negative integer.
    :param mu: The average rate of success (events occurring), must be non-negative real number.
    :return: The probability of k successes occurring in a given time period or space.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    return (mu ** k * _exp(-mu)) / _factorial(k)


def rayleigh_pdf(x, sigma):
    """
    Introduction
    ==========
    Rayleigh probability density function

    Example
    =========
    >>> rayleigh_pdf(1, 1)
    0.6065306597126334
    >>>
    :param x: The value at which to evaluate the PDF. It must be non-negative.
    :param sigma: The scale parameter of the distribution, must be positive.
    :return: The probability density at x for the Rayleigh distribution with scale parameter sigma.
    """
    if sigma <= 0:
        raise ValueError("Scale parameter sigma must be positive")
    if x < 0:
        raise ValueError("x must be non-negative")
    return x / (sigma ** 2) * _exp(-x ** 2 / (2 * sigma ** 2))


def t_pdf(x, df=1):
    """
    Introduction
    ==========
    Student's t probability density function

    Example
    ==========
    >>> t_pdf(1, 2)
    0.1924500897298753
    >>>
    :param x: The value at which to evaluate the PDF.
    :param df: The degrees of freedom, must be positive.
    :return: The probability density at x for the Student's t distribution with df degrees of freedom.
    """
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive")
    return _gamma((df + 1) / 2) / (_sqrt(df * 3.141592653589793)
                                   * _gamma(df / 2)) * (1 + x ** 2 / df) ** (-(df + 1) / 2)


def uniform_pdf(x, loc=0, scale=1):
    """
    Introduction
    ==========
    Continuous uniform probability density function

    Example
    ==========
    >>> uniform_pdf(0.5, 0, 1)
    1.0
    >>>
    :param x: The value at which to evaluate the PDF.
    :param loc: The location parameter, the lower bound of the distribution.
    :param scale: The scale parameter, the width of the interval over which the distribution is defined,
        must be positive.
    :return: The probability density at x for the uniform distribution between loc and loc + scale.
    """
    if scale <= 0:
        raise ValueError("Scale parameter must be positive")
    return 1 / scale if loc <= x < loc + scale else 0


def uniform_cdf(x, loc=0, scale=1):
    """
    Introduction
    ==========
    Continuous uniform cumulative distribution function

    Example
    ==========
    >>> uniform_cdf(0.5, 0, 1)
    0.5
    >>>
    :param x: The value at which to evaluate the CDF.
    :param loc: The location parameter, the lower bound of the distribution.
    :param scale: The scale parameter, the width of the interval over which the distribution is defined,
        must be positive.
    :return: The cumulative probability at x for the uniform distribution between loc and loc + scale.
    """
    if scale <= 0:
        raise ValueError("Scale parameter must be positive")
    return 0 if x < loc else (x - loc) / scale if loc <= x <= loc + scale else 1


def vonmises_pdf(x, mu, kappa):
    """
    Introduction
    ==========
    Von Mises probability density function

    Example
    ==========
    >>> vonmises_pdf(0.7853981633974483, 1.5707963267948966, 2)
    0.28717685154709915
    >>>
    :param x: The value at which to evaluate the PDF, must be in the range [0, 2*pi).
    :param mu: The location parameter, representing the mean direction, must be in the range [0, 2*pi).
    :param kappa: The concentration parameter, must be positive.
    :return: The probability density at x for the von Mises distribution with the given parameters.
    """
    from .maths import bessel_i0
    if not 0 <= mu < 6.283185307179586:
        raise ValueError("mu must be between 0 and 2 * pi")
    if not 0 <= x < 6.283185307179586:
        raise ValueError("x must be between 0 and 2 * pi")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    return _exp(kappa * _cos(x - mu)) / 6.283185307179586 / bessel_i0(kappa)


def weibull_max_pdf(x, c=1, scale=1, loc=0):
    """
    Introduction
    ==========
    Weibull maximum probability density function

    Example
    ==========
    >>> weibull_max_pdf(0, 1, 1, 0)
    1.0
    >>>
    :param x: The value at which to evaluate the PDF, must be less than or equal to loc.
    :param c: The shape parameter, must be positive.
    :param scale: The scale parameter, must be positive.
    :param loc: The location parameter, the upper bound of the distribution.
    :return: The probability density at x for the Weibull distribution with the given parameters.
    """
    if c <= 0 or scale <= 0:
        raise ValueError("Shape and scale parameters must be positive")
    if x > loc:
        return 0
    return (c / scale) * ((loc - x) / scale) ** (c - 1) * _exp(-((loc - x) / scale) ** c)


def weibull_min_pdf(x, c=1, scale=1, loc=0):
    """
    Introduction
    ==========
    Weibull minimum probability density function

    Example
    ==========
    >>> weibull_min_pdf(1, 1, 1, 0)
    0.36787944117144233
    >>>
    :param x: The value at which to evaluate the PDF, must be greater than or equal to loc.
    :param c: The shape parameter, must be positive.
    :param scale: The scale parameter, must be positive.
    :param loc: The location parameter, the lower bound of the distribution.
    :return: The probability density at x for the Weibull distribution with the given parameters.
    """
    if c <= 0 or scale <= 0:
        raise ValueError("Shape and scale parameters must be positive")
    if x < loc:
        return 0
    return (c / scale) * ((x - loc) / scale) ** (c - 1) * _exp(-((x - loc) / scale) ** c)


def zipf_pmf(k, s, n):
    """
    Introduction
    ==========
    Zipf probability mass function

    Example
    =========
    >>> zipf_pmf(1, 1, 10)
    0.34141715214740553
    >>>
    :param k: The rank at which to evaluate the PMF. It must be an integer between 1 and n, inclusive.
    :param s: The exponent parameter of the distribution, must be positive.
    :param n: The number of elements in the distribution, must be a positive integer.
    :return: The probability mass at rank k for the Zipf distribution with exponent parameter s
        and number of elements n.
    """
    if s <= 0:
        raise ValueError("Exponent parameter s must be positive")
    if n <= 0 or not isinstance(n, int):
        raise ValueError("Number of elements n must be a positive integer")
    if k <= 0 or k > n:
        raise ValueError("Rank k must be an integer between 1 and n, inclusive")
    return 1 / k ** s / sum([1 / (i ** s) for i in range(1, n + 1)])


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
