from .confs import config


class Polynomial:
    def __init__(self, terms=None):
        if terms is None:
            self.terms = []
        else:
            try:
                self.terms = sorted([tuple(item) if len(item) == 2 else None for item in terms if item[1] != 0])
            except TypeError:
                raise TypeError("The input polynomial must be a sequence consisting of degree-coefficient pairs")

    def add_term(self, degree, coefficient):
        if coefficient == 0:
            return

        def binary_search(terms, deg):
            low, high = 0, len(terms) - 1
            while low <= high:
                mid = (low + high) // 2
                if terms[mid][0] == deg:
                    return mid
                elif terms[mid][0] < deg:
                    low = mid + 1
                else:
                    high = mid - 1
            return low

        insert_index = binary_search(self.terms, degree)
        if insert_index < len(self.terms) and self.terms[insert_index][0] == degree:
            try:
                self.terms[insert_index] = (degree, self.terms[insert_index][1] + coefficient)
            except OverflowError:
                self.terms[insert_index] = (degree, int(self.terms[insert_index][1]) + int(coefficient))
        else:
            self.terms.insert(insert_index, (degree, coefficient))

    def is_zero(self):
        return not self.terms

    def gcd(self, other):
        temp = self
        while not other.is_zero():
            temp, other = other, temp % other
        return temp

    def lcm(self, other):
        gcd_poly = self.gcd(other)
        if gcd_poly.is_zero():
            raise ValueError("Cannot compute LCM of polynomials with zero GCD")
        return self * other // gcd_poly

    def roots(self, real_roots=True, tolerance=1e-15):
        from .equations import poly_eq
        all_roots = poly_eq(self.coeffs(True))
        if real_roots:
            return [(1 if root.real > 0 else -1) * abs(root) for root in all_roots if abs(root.imag) < tolerance]
        return all_roots

    def evaluate(self, x):
        return sum([coefficient * x ** degree for degree, coefficient in self.terms])

    def degs(self, reverse=False):
        terms = reversed(self.terms) if reverse else self.terms
        return [deg for deg, _ in terms]

    def coeffs(self, reverse=False):
        max_deg = self.degree()
        coeffs = [0] * (max_deg + 1)
        for deg, coeff in self.terms:
            coeffs[deg] = coeff
        if reverse:
            coeffs.reverse()
        return coeffs

    def sqrt(self):
        deg, coeff = self.terms[-1]
        new = deg // 2, coeff ** 0.5
        root = Polynomial([new])
        temp = Polynomial([new])
        rem = self - root * root
        for _ in range(deg):
            temp.add_term(*temp.terms[0])
            try:
                root.add_term(*(rem // temp).terms[-1])
            except IndexError:
                break
            temp.add_term(*root.terms[0])
            rem = rem - temp * Polynomial([temp.terms[0]])
        return root, rem

    def remove0terms(self):
        self.terms = [(deg, coeff) for deg, coeff in self.terms if coeff != 0]

    def degree(self):
        if self.terms:
            return self.terms[-1][0]
        return -1

    def lead_coeff(self):
        if self.terms:
            return self.terms[-1][1]
        return 0

    def deriv(self):
        derivative_terms = [(deg - 1, deg * coeff) for deg, coeff in self.terms if deg > 0]
        return Polynomial(derivative_terms)

    def integ(self, constant=0):
        integrated_terms = [(deg + 1, coeff / (deg + 1)) for deg, coeff in self.terms]
        if constant != 0:
            integrated_terms.append((0, constant))
        return Polynomial(integrated_terms)

    def latex(self):
        return self.__repr__(True)

    def __repr__(self, use_latex=False):
        from .chars import int2superscript
        use_latex = config.use_latex or use_latex
        use_unicode = config.use_unicode
        if not self.terms:
            return "0"
        result = ""
        for i, (degree, coefficient) in enumerate(self.terms):
            if i != 0:
                if coefficient < 0:
                    result += " - "
                else:
                    result += " + "
            elif coefficient < 0:
                result += "-"
            if degree == 0:
                result += str(abs(coefficient))
            elif abs(coefficient) != 1:
                result += str(abs(coefficient))
            if degree > 0:
                if degree > 1:
                    if use_latex:
                        result += "x^{" + str(degree) + "}"
                    elif use_unicode:
                        result += "x" + int2superscript(str(degree))
                    else:
                        result += "x^" + str(degree)
                else:
                    result += "x"
        return result

    def __pos__(self):
        return Polynomial([(degree, coefficient) for degree, coefficient in self.terms])

    def __neg__(self):
        return Polynomial([(degree, -coefficient) for degree, coefficient in self.terms])

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Polynomial([(deg, coeff + other if deg == 0 else coeff) for deg, coeff in self.terms])
        result = Polynomial()
        i, j = 0, 0
        while i < len(self.terms) or j < len(other.terms):
            if i == len(self.terms):
                result.add_term(other.terms[j][0], other.terms[j][1])
                j += 1
            elif j == len(other.terms):
                result.add_term(self.terms[i][0], self.terms[i][1])
                i += 1
            else:
                degree_self, coefficient_self = self.terms[i]
                degree_other, coefficient_other = other.terms[j]
                if degree_self < degree_other:
                    result.add_term(degree_self, coefficient_self)
                    i += 1
                elif degree_self > degree_other:
                    result.add_term(degree_other, coefficient_other)
                    j += 1
                else:
                    try:
                        result.add_term(degree_self, coefficient_self + coefficient_other)
                    except OverflowError:
                        result.add_term(degree_self, int(coefficient_self) + int(coefficient_other))
                    i += 1
                    j += 1
        return result

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Polynomial([(deg, coeff - other if deg == 0 else coeff) for deg, coeff in self.terms])
        negative_other = -other
        return self + negative_other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Polynomial([(deg, coeff * other) for deg, coeff in self.terms])
        result_terms = {}
        for degree_self, coeff_self in self.terms:
            for degree_other, coeff_other in other.terms:
                degree_product = degree_self + degree_other
                coeff_product = coeff_self * coeff_other
                if degree_product in result_terms:
                    result_terms[degree_product] = result_terms[degree_product] + coeff_product
                else:
                    result_terms[degree_product] = coeff_product
        return Polynomial(sorted(result_terms.items()))

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ValueError("Cannot divide by zero")
            return Polynomial([(deg, coeff / other) for deg, coeff in self.terms])
        if not other.terms or other.lead_coeff() == 0:
            raise ValueError("Cannot divide by zero polynomial")
        quotient = Polynomial()
        remainder = Polynomial(self.terms)
        other_degree = other.degree()
        other_lead_coeff = other.lead_coeff()
        while remainder.degree() >= other_degree:
            quotient_degree = remainder.degree() - other_degree
            try:
                quotient_coefficient = (remainder.lead_coeff() / other_lead_coeff)
            except OverflowError:
                quotient_coefficient = (remainder.lead_coeff() // other_lead_coeff)
            quotient_term = Polynomial([(quotient_degree, quotient_coefficient)])
            quotient.add_term(quotient_degree, quotient_coefficient)
            remainder = remainder - quotient_term * other
        return quotient, remainder

    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            return Polynomial([(deg, coeff // other) for deg, coeff in self.terms])
        return (self / other)[0]

    def __mod__(self, other):
        if isinstance(other, (int, float)):
            return Polynomial([(deg, coeff % other) for deg, coeff in self.terms])
        return (self / other)[1]

    def __divmod__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ValueError("Cannot divide or modulo by zero")
            int_part_terms = []
            mod_part_terms = []
            for deg, coeff in self.terms:
                int_coeff, mod_coeff = divmod(coeff, other)
                if int_coeff != 0 or deg == 0:
                    int_part_terms.append((deg, int_coeff))
                if mod_coeff != 0:
                    mod_part_terms.append((deg, mod_coeff))
            int_part = Polynomial(int_part_terms)
            mod_part = Polynomial(mod_part_terms)
            return int_part, mod_part
        else:
            return self / other

    def __int__(self):
        self.terms = [(degree, int(coefficient)) for degree, coefficient in self.terms]
        return 0

    def __float__(self):
        self.terms = [(degree, float(coefficient)) for degree, coefficient in self.terms]
        return 0.0

    def __round__(self, n=None):
        return Polynomial([(degree, round(coefficient, n)) for degree, coefficient in self.terms])

    def __pow__(self, power, modulo=None):
        if power == 0:
            return Polynomial([(0, 1)])
        result = Polynomial([(0, 1)])
        for _ in range(power):
            result = result * self
        if modulo:
            result = result % modulo
        return result

    def stationaries(self, tolerance=1e-15):
        derivative = self.deriv()
        return derivative.roots(tolerance=tolerance)

    def saddles(self, tolerance=1e-15):
        first_derivative = self.deriv()
        second_derivative = first_derivative.deriv()
        stationary_points = self.stationaries(tolerance=tolerance)
        saddle_points = []
        for point in stationary_points:
            if abs(second_derivative.evaluate(point)) > tolerance and first_derivative.evaluate(
                    point) * second_derivative.evaluate(point) < 0:
                saddle_points.append(point)
        return saddle_points

    def minima(self, tolerance=1e-15):
        second_derivative = self.deriv().deriv()
        stationary_points = self.stationaries(tolerance=tolerance)
        minimums = []
        for point in stationary_points:
            if second_derivative.evaluate(point) > 0:
                minimums.append(point)
        return minimums

    def maxima(self, tolerance=1e-15):
        second_derivative = self.deriv().deriv()
        stationary_points = self.stationaries(tolerance=tolerance)
        maximums = []
        for point in stationary_points:
            if second_derivative.evaluate(point) < 0:
                maximums.append(point)
        return maximums

    def inflections(self, tolerance=1e-15):
        second_derivative = self.deriv().deriv()
        third_derivative = second_derivative.deriv()
        potential_inflections = second_derivative.roots(tolerance=tolerance)
        inflection_points = [x for x in potential_inflections if abs(third_derivative.evaluate(x)) > tolerance]
        return inflection_points


def poly(terms=None):
    return Polynomial(terms)


def from_coeffs(coeffs):
    return Polynomial([(degree, coefficient) for degree, coefficient in enumerate(coeffs) if coefficient != 0])


def from_coords(coords):
    from .regression import poly_reg
    return from_coeffs(reversed(poly_reg(*zip(*coords))))


def legpoly(n, single=True):
    x = Polynomial([(1, 1)])
    p = [Polynomial([(0, 1)]), x]
    for i in range(1, n):
        m = Polynomial([(0, (2 * i + 1) / (i + 1))]) * x * p[i] - Polynomial([(0, i / (i + 1))]) * p[i - 1]
        p.append(m)
    return p[n] if single else p


def leggauss(n):
    polynomial = legpoly(n)
    roots = polynomial.roots(False)
    weights = []
    for node in roots:
        factor1 = 1 - node ** 2
        derivative_poly = polynomial.deriv()
        derivative_value = derivative_poly.evaluate(node)
        weight = 2 / (factor1 * derivative_value ** 2)
        weights.append(weight)
    return roots, weights


def chebpoly(n, single=True):
    result = [Polynomial([(0, 1)]), Polynomial([(1, 1)])]
    multiple = Polynomial([(1, 2)])
    for _ in range(n - 1):
        result.append(multiple * result[-1] - result[-2])
    return result[n] if single else result


def chebgauss(n):
    from math import cos
    pi_over_n = 3.141592653589793 / n
    pi_over_2n = pi_over_n / 2
    roots = [cos((2 * i - 1) * pi_over_2n) for i in range(1, n + 1)]
    weights = [pi_over_n for _ in range(n)]
    return roots, weights


def lagpoly(n, single=True):
    result = [Polynomial([(0, 1)]), Polynomial([(0, 1), (1, -1)])]
    for i in range(2, n + 1):
        result.append((Polynomial([(0, 2 * i - 1), (1, -1)]) * result[-1] - Polynomial([(0, i - 1)]) * result[-2]) / i)
    return result[n] if single else result


def laggauss(n):
    polynomial, next_polynomial = lagpoly(n + 1, False)[-2:]
    roots = polynomial.roots()
    weights = []
    const = (n + 1) ** 2
    for node in roots:
        next_polynomial_value = next_polynomial.evaluate(node)
        weight = node / const * next_polynomial_value ** 2
        weights.append(weight)
    return roots, weights
