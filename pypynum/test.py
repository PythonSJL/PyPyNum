"""
A Code Test File
"""

print("""\x1b[92m
from pypynum import (arrays, geoms, hypcmpnms, logics, matrices, multiprec, special, vectors,
                     ciphers, consts, equations, maths, plotting, random, regs, tools)
\x1b[m""")
print("...")

from . import (arrays, geoms, hypcmpnms, logics, matrices, multiprec, special, vectors,
               ciphers, consts, equations, maths, plotting, random, regs, tools)

# arrays
print("""\x1b[92m
print(arrays.array())
print(arrays.array([1, 2, 3, 4, 5, 6, 7, 8]))
print(arrays.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
print(arrays.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
\x1b[m""")
print("\"\"\"")
print(arrays.array())
print(arrays.array([1, 2, 3, 4, 5, 6, 7, 8]))
print(arrays.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
print(arrays.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
print("\"\"\"")

# geoms
print("""\x1b[92m
triangle = geoms.Triangle((0, 0), (2, 2), (3, 0))
print(triangle.perimeter())
print(triangle.area())
print(triangle.centroid())
\x1b[m""")
print("\"\"\"")
triangle = geoms.Triangle((0, 0), (2, 2), (3, 0))
print(triangle.perimeter())
print(triangle.area())
print(triangle.centroid())
print("\"\"\"")

# hypcmpnms
print("""\x1b[92m
q0 = hypcmpnms.quat(1, 2, 3, 4)
q1 = hypcmpnms.quat(5, 6, 7, 8)
print(q0)
print(q1)
print(q0 + q1)
print(q0 * q1)
print(q0.inverse())
print(q1.conjugate())
\x1b[m""")
print("\"\"\"")
q0 = hypcmpnms.quat(1, 2, 3, 4)
q1 = hypcmpnms.quat(5, 6, 7, 8)
print(q0)
print(q1)
print(q0 + q1)
print(q0 * q1)
print(q0.inverse())
print(q1.conjugate())
print("\"\"\"")

# logics
print("""\x1b[92m
a, b, c = 1, 1, 1
adder0, adder1 = logics.HalfAdder("alpha", a, b), logics.HalfAdder("beta", c, None)
xor0 = logics.XOR("alpha")
ff0, ff1 = logics.DFF("alpha"), logics.DFF("beta")
xor0.set_order0(1)
xor0.set_order1(1)
logics.connector(adder0, adder1)
logics.connector(adder0, xor0)
logics.connector(adder1, xor0)
logics.connector(adder1, ff0)
logics.connector(xor0, ff1)
print("sum: {}, carry: {}".format(ff0.out(), ff1.out()))
\x1b[m""")
print("\"\"\"")
a, b, c = 1, 1, 1
adder0, adder1 = logics.HalfAdder("alpha", a, b), logics.HalfAdder("beta", c, None)
xor0 = logics.XOR("alpha")
ff0, ff1 = logics.DFF("alpha"), logics.DFF("beta")
xor0.set_order0(1)
xor0.set_order1(1)
logics.connector(adder0, adder1)
logics.connector(adder0, xor0)
logics.connector(adder1, xor0)
logics.connector(adder1, ff0)
logics.connector(xor0, ff1)
print("sum: {}, carry: {}".format(ff0.out(), ff1.out()))
print("\"\"\"")

# matrices
print("""\x1b[92m
m0 = matrices.mat([[1, 2], [3, 4]])
m1 = matrices.mat([[5, 6], [7, 8]])
print(m0)
print(m1)
print(m0 + m1)
print(m0 @ m1)
print(m0.inv())
print(m1.rank())
\x1b[m""")
print("\"\"\"")
m0 = matrices.mat([[1, 2], [3, 4]])
m1 = matrices.mat([[5, 6], [7, 8]])
print(m0)
print(m1)
print(m0 + m1)
print(m0 @ m1)
print(m0.inv())
print(m1.rank())
print("\"\"\"")

# multiprec
print("""\x1b[92m
mp_complex1 = multiprec.MPComplex("1.4142135623730950488016887242096980785696718753769",
                                  "2.7182818284590452353602874713527", sigfigs=30)
mp_complex2 = multiprec.MPComplex("1.7320508075688772935274463415059",
                                  "3.141592653589793238462643383279502884197169399375105820974944", sigfigs=40)
modulus = mp_complex1.modulus(sigfigs=25)
print("Modulus of the complex1:", modulus)
sqrt_complex = mp_complex2.sqrt()
print("Square root of the complex2:", sqrt_complex)
power_result = mp_complex1 ** mp_complex2
print("Power of complex1 raised to complex2:", power_result)
euler_gamma = multiprec.mp_euler_gamma(sigfigs=45)
print("Value of Euler's gamma constant:", euler_gamma)
log_2 = multiprec.mp_log(2, 10, sigfigs=50)
print("Logarithm of 2 (base 10):", log_2)
exp_e_squared = multiprec.mp_exp(multiprec.mp_e() ** 2, sigfigs=20)
print("Value of exp(e^2):", exp_e_squared)
\x1b[m""")
print("\"\"\"")
mp_complex1 = multiprec.MPComplex("1.4142135623730950488016887242096980785696718753769",
                                  "2.7182818284590452353602874713527", sigfigs=30)
mp_complex2 = multiprec.MPComplex("1.7320508075688772935274463415059",
                                  "3.141592653589793238462643383279502884197169399375105820974944", sigfigs=40)
modulus = mp_complex1.modulus(sigfigs=25)
print("Modulus of the complex1:", modulus)
sqrt_complex = mp_complex2.sqrt()
print("Square root of the complex2:", sqrt_complex)
power_result = mp_complex1 ** mp_complex2
print("Power of complex1 raised to complex2:", power_result)
euler_gamma = multiprec.mp_euler_gamma(sigfigs=45)
print("Value of Euler's gamma constant:", euler_gamma)
log_2 = multiprec.mp_log(2, 10, sigfigs=50)
print("Logarithm of 2 (base 10):", log_2)
exp_e_squared = multiprec.mp_exp(multiprec.mp_e() ** 2, sigfigs=20)
print("Value of exp(e^2):", exp_e_squared)
print("\"\"\"")

# special
print("""\x1b[92m
print("Bessel Function of the first kind, order 0 at x=1:", special.besselj0(1))
print("Modified Bessel function of the first kind, order 1 at x=1:", special.besseli1(1))
print("Hypergeometric function 0F1 at z=0.5 with b0=1:", special.hyp0f1(1, 0.5))
print("Hypergeometric function 1F1 at z=1 with a0=1, b0=1:", special.hyp1f1(1, 1, 1))
print("q-Pochhammer Symbol with a=2+1j, q=0.5+0.1j, n=2+1j:", special.qpochhammer(2 + 1j, 0.5 + 0.1j, 2 + 1j))
print("q-Gamma Function at n=2 with q=0.5+0.1j:", special.qgamma(2, 0.5 + 0.1j))
\x1b[m""")
print("\"\"\"")
print("Bessel Function of the first kind, order 0 at x=1:", special.besselj0(1))
print("Modified Bessel function of the first kind, order 1 at x=1:", special.besseli1(1))
print("Hypergeometric function 0F1 at z=0.5 with b0=1:", special.hyp0f1(1, 0.5))
print("Hypergeometric function 1F1 at z=1 with a0=1, b0=1:", special.hyp1f1(1, 1, 1))
print("q-Pochhammer Symbol with a=2+1j, q=0.5+0.1j, n=2+1j:", special.qpochhammer(2 + 1j, 0.5 + 0.1j, 2 + 1j))
print("q-Gamma Function at n=2 with q=0.5+0.1j:", special.qgamma(2, 0.5 + 0.1j))
print("\"\"\"")

# ciphers
print("""\x1b[92m
string = "PyPyNum"
encrypted = ciphers.caesar(string, 10)
print(string)
print(encrypted)
print(ciphers.caesar(encrypted, 10, decrypt=True))
encrypted = ciphers.vigenere(string, "ciphers")
print(string)
print(encrypted)
print(ciphers.vigenere(encrypted, "ciphers", decrypt=True))
encrypted = ciphers.morse(string)
print(string)
print(encrypted)
print(ciphers.morse(encrypted, decrypt=True))
\x1b[m""")
print("\"\"\"")
string = "PyPyNum"
encrypted = ciphers.caesar(string, 10)
print(string)
print(encrypted)
print(ciphers.caesar(encrypted, 10, decrypt=True))
encrypted = ciphers.vigenere(string, "ciphers")
print(string)
print(encrypted)
print(ciphers.vigenere(encrypted, "ciphers", decrypt=True))
encrypted = ciphers.morse(string)
print(string)
print(encrypted)
print(ciphers.morse(encrypted, decrypt=True))
print("\"\"\"")

# vectors
print("""\x1b[92m
v0 = vectors.vec([1, 2, 3, 4])
v1 = vectors.vec([5, 6, 7, 8])
print(v0)
print(v1)
print(v0 + v1)
print(v0 @ v1)
print(v0.normalize())
print(v1.angles())
\x1b[m""")
print("\"\"\"")
v0 = vectors.vec([1, 2, 3, 4])
v1 = vectors.vec([5, 6, 7, 8])
print(v0)
print(v1)
print(v0 * v1)
print(v0 @ v1)
print(v0.normalize())
print(v1.angles())
print("\"\"\"")

# consts
print("""\x1b[92m
print(consts.TB)
print(consts.e)
print(consts.h)
print(consts.phi)
print(consts.pi)
print(consts.tera)
\x1b[m""")
print("\"\"\"")
print(consts.TB)
print(consts.e)
print(consts.h)
print(consts.phi)
print(consts.pi)
print(consts.tera)
print("\"\"\"")

# equations
print("""\x1b[92m
p = [1, -2, -3, 4]
m = [
    [
        [1, 2, 3],
        [6, 10, 12],
        [7, 16, 9]
    ],
    [-1, -2, -3]
]
print(equations.poly_eq(p))
print(equations.lin_eq(*m))
\x1b[m""")
print("\"\"\"")
p = [1, -2, -3, 4]
m = [
    [
        [1, 2, 3],
        [6, 10, 12],
        [7, 16, 9]
    ],
    [-1, -2, -3]
]
print(equations.poly_eq(p))
print(equations.lin_eq(*m))
print("\"\"\"")

# maths
print("""\x1b[92m
print(maths.cot(consts.pi / 3))
print(maths.gamma(1.5))
print(maths.pi(1, 10, lambda x: x ** 2))
print(maths.product([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))
print(maths.sigma(1, 10, lambda x: x ** 2))
print(maths.var([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))
\x1b[m""")
print("\"\"\"")
print(maths.cot(consts.pi / 3))
print(maths.gamma(1.5))
print(maths.pi(1, 10, lambda x: x ** 2))
print(maths.product([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))
print(maths.sigma(1, 10, lambda x: x ** 2))
print(maths.var([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))
print("\"\"\"")

# plotting
print("""\x1b[92m
plt = plotting.unary(lambda x: x ** 2, top=10, bottom=0, character="+")
print(plt)
print(plotting.binary(lambda x, y: x ** 2 + y ** 2 - 10, right=10, left=0, compare="<=", basic=plotting.change(plt)))
print(plotting.c_unary(lambda x: x ** x, right=2, left=-2, top=2, bottom=-2, complexity=20, character="-"))
\x1b[m""")
print("\"\"\"")
plt = plotting.unary(lambda x: x ** 2, top=10, bottom=0, character="+")
print(plt)
print(plotting.binary(lambda x, y: x ** 2 + y ** 2 - 10, right=10, left=0, compare="<=", basic=plotting.change(plt)))
print(plotting.c_unary(lambda x: x ** x, right=2, left=-2, top=2, bottom=-2, complexity=20, character="-"))
print("\"\"\"")

# random
print("""\x1b[92m
print(random.gauss(0, 1, [2, 3, 4]))
print(random.rand([2, 3, 4]))
print(random.randint(0, 9, [2, 3, 4]))
print(random.uniform(0, 9, [2, 3, 4]))
\x1b[m""")
print("\"\"\"")
print(random.gauss(0, 1, [2, 3, 4]))
print(random.rand([2, 3, 4]))
print(random.randint(0, 9, [2, 3, 4]))
print(random.uniform(0, 9, [2, 3, 4]))
print("\"\"\"")

# regs
print("""\x1b[92m
print(regs.lin_reg(list(range(5)), [2, 4, 6, 7, 8]))
print(regs.par_reg(list(range(5)), [2, 4, 6, 7, 8]))
print(regs.poly_reg(list(range(5)), [2, 4, 6, 7, 8], 4))
\x1b[m""")
print("\"\"\"")
print(regs.lin_reg(list(range(5)), [2, 4, 6, 7, 8]))
print(regs.par_reg(list(range(5)), [2, 4, 6, 7, 8]))
print(regs.poly_reg(list(range(5)), [2, 4, 6, 7, 8], 4))
print("\"\"\"")

# tools
print("""\x1b[92m
print(tools.classify([1, 2.3, 4 + 5j, "string", list, True, 3.14, False, tuple, tools]))
print(tools.dedup(["Python", 6, "NumPy", int, "PyPyNum", 9, "pypynum", "NumPy", 6, True]))
print(tools.frange(0, 3, 0.4))
print(tools.linspace(0, 2.8, 8))
\x1b[m""")
print("\"\"\"")
print(tools.classify([1, 2.3, 4 + 5j, "string", list, True, 3.14, False, tuple, tools]))
print(tools.dedup(["Python", 6, "NumPy", int, "PyPyNum", 9, "pypynum", "NumPy", 6, True]))
print(tools.frange(0, 3, 0.4))
print(tools.linspace(0, 2.8, 8))
print("\"\"\"")

for _ in tuple(globals()):
    if _[0] != "_":
        del globals()[_]
del _

print("""\x1b[91m
# Tip:
# The test has been successfully passed and ended.
# These tests are only part of the functionality of this package.
# More features need to be explored and tried by yourself!
\x1b[m""")
