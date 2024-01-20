"""
A Code Test File
"""

print("""\033[92m
>>> from pypynum import (Array, Geometry, Logic, Matrix, Quaternion, Symbolics, Tensor, Vector,
                         cipher, constants, equations, mathematics, plotting, random, regression, tools)
\033[m""")
print("...")

from . import (Array, Geometry, Logic, Matrix, Quaternion, Symbolics, Tensor, Vector,
               cipher, constants, equations, mathematics, plotting, random, regression, tools)

# Array
print("""\033[92m
>>> print(Array.array())
>>> print(Array.array([1, 2, 3, 4, 5, 6, 7, 8]))
>>> print(Array.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
>>> print(Array.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
\033[m""")
print(Array.array())
print(Array.array([1, 2, 3, 4, 5, 6, 7, 8]))
print(Array.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
print(Array.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

# Geometry
print("""\033[92m
>>> triangle = Geometry.Triangle((0, 0), (2, 2), (3, 0))
>>> print(triangle.perimeter())
>>> print(triangle.area())
>>> print(triangle.centroid())
\033[m""")
triangle = Geometry.Triangle((0, 0), (2, 2), (3, 0))
print(triangle.perimeter())
print(triangle.area())
print(triangle.centroid())

# Logic
print("""\033[92m
>>> a, b, c = 1, 1, 1
>>> adder0, adder1 = Logic.HalfAdder("alpha", a, b), Logic.HalfAdder("beta", c, None)
>>> xor0 = Logic.XOR("alpha")
>>> ff0, ff1 = Logic.DFF("alpha"), Logic.DFF("beta")
>>> xor0.set_order0(1)
>>> xor0.set_order1(1)
>>> Logic.connector(adder0, adder1)
>>> Logic.connector(adder0, xor0)
>>> Logic.connector(adder1, xor0)
>>> Logic.connector(adder1, ff0)
>>> Logic.connector(xor0, ff1)
>>> print("sum: {}, carry: {}".format(ff0.out(), ff1.out()))
\033[m""")
a, b, c = 1, 1, 1
adder0, adder1 = Logic.HalfAdder("alpha", a, b), Logic.HalfAdder("beta", c, None)
xor0 = Logic.XOR("alpha")
ff0, ff1 = Logic.DFF("alpha"), Logic.DFF("beta")
xor0.set_order0(1)
xor0.set_order1(1)
Logic.connector(adder0, adder1)
Logic.connector(adder0, xor0)
Logic.connector(adder1, xor0)
Logic.connector(adder1, ff0)
Logic.connector(xor0, ff1)
print("sum: {}, carry: {}".format(ff0.out(), ff1.out()))

# Matrix
print("""\033[92m
>>> m0 = Matrix.mat([[1, 2], [3, 4]])
>>> m1 = Matrix.mat([[5, 6], [7, 8]])
>>> print(m0)
>>> print(m1)
>>> print(m0 + m1)
>>> print(m0 @ m1)
>>> print(m0.inv())
>>> print(m1.rank())
\033[m""")
m0 = Matrix.mat([[1, 2], [3, 4]])
m1 = Matrix.mat([[5, 6], [7, 8]])
print(m0)
print(m1)
print(m0 + m1)
print(m0 @ m1)
print(m0.inv())
print(m1.rank())

# Quaternion
print("""\033[92m
>>> q0 = Quaternion.quat(1, 2, 3, 4)
>>> q1 = Quaternion.quat(5, 6, 7, 8)
>>> print(q0)
>>> print(q1)
>>> print(q0 + q1)
>>> print(q0 * q1)
>>> print(q0.inverse())
>>> print(q1.conjugate())
\033[m""")
q0 = Quaternion.quat(1, 2, 3, 4)
q1 = Quaternion.quat(5, 6, 7, 8)
print(q0)
print(q1)
print(q0 + q1)
print(q0 * q1)
print(q0.inverse())
print(q1.conjugate())

# Symbolics
print("""\033[92m
>>> print(Symbolics.basic)
>>> print(Symbolics.english)
>>> print(Symbolics.greek)
>>> print(Symbolics.interpreter("-(10+a-(3.14+b0)*(-5))**(-ζn1-2.718/mΣ99)//9"))
\033[m""")
print(Symbolics.basic)
print(Symbolics.english)
print(Symbolics.greek)
print(Symbolics.interpreter("-(10+a-(3.14+b0)*(-5))**(-ζn1-2.718/mΣ99)//9"))

# Tensor
print("""\033[92m
>>> t0 = Tensor.ten([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
>>> t1 = Tensor.ten([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
>>> print(t0)
>>> print(t1)
>>> print(t0 + t1)
>>> print(t0 @ t1)
\033[m""")
t0 = Tensor.ten([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
t1 = Tensor.ten([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
print(t0)
print(t1)
print(t0 + t1)
print(t0 @ t1)

# cipher
print("""\033[92m
>>> string = "PyPyNum"
>>> encrypted = cipher.dna(string)
>>> print(string)
>>> print(encrypted)
>>> print(cipher.dna(encrypted, decrypt=True))
\033[m""")
string = "PyPyNum"
encrypted = cipher.dna(string)
print(string)
print(encrypted)
print(cipher.dna(encrypted, decrypt=True))

# Vector
print("""\033[92m
v0 = Vector.vec([1, 2, 3, 4])
v1 = Vector.vec([5, 6, 7, 8])
print(v0)
print(v1)
print(v0 + v1)
print(v0 @ v1)
print(v0.normalize())
print(v1.angles())
\033[m""")
v0 = Vector.vec([1, 2, 3, 4])
v1 = Vector.vec([5, 6, 7, 8])
print(v0)
print(v1)
print(v0 * v1)
print(v0 @ v1)
print(v0.normalize())
print(v1.angles())

# constants
print("""\033[92m
>>> print(constants.TB)
>>> print(constants.e)
>>> print(constants.h)
>>> print(constants.phi)
>>> print(constants.pi)
>>> print(constants.tera)
\033[m""")
print(constants.TB)
print(constants.e)
print(constants.h)
print(constants.phi)
print(constants.pi)
print(constants.tera)

# equations
print("""\033[92m
>>> p = [1, -2, -3, 4]
>>> m = [
    [
        [1, 2, 3],
        [6, 10, 12],
        [7, 16, 9]
    ],
    [-1, -2, -3]
]
>>> print(equations.pe(p))
>>> print(equations.mles(*m))
\033[m""")
p = [1, -2, -3, 4]
m = [
    [
        [1, 2, 3],
        [6, 10, 12],
        [7, 16, 9]
    ],
    [-1, -2, -3]
]
print(equations.pe(p))
print(equations.mles(*m))

# mathematics
print("""\033[92m
>>> print(mathematics.cot(constants.pi / 3))
>>> print(mathematics.gamma(1.5))
>>> print(mathematics.pi(1, 10, lambda x: x ** 2))
>>> print(mathematics.product([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))
>>> print(mathematics.sigma(1, 10, lambda x: x ** 2))
>>> print(mathematics.var([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))
\033[m""")
print(mathematics.cot(constants.pi / 3))
print(mathematics.gamma(1.5))
print(mathematics.pi(1, 10, lambda x: x ** 2))
print(mathematics.product([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))
print(mathematics.sigma(1, 10, lambda x: x ** 2))
print(mathematics.var([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))

# plotting
print("""\033[92m
>>> plt = plotting.unary(lambda x: x ** 2, top=10, bottom=0, character="+")
>>> print(plt)
>>> print(plotting.binary(lambda x, y: x ** 2 + y ** 2 - 10, right=10, left=0, compare="<=", basic=plotting.change(plt)))
>>> print(plotting.c_unary(lambda x: x ** x, start=-10, end=10, interval=100, right=2, left=-2, top=2, bottom=-2, complexity=20, character="-"))
\033[m""")
plt = plotting.unary(lambda x: x ** 2, top=10, bottom=0, character="+")
print(plt)
print(plotting.binary(lambda x, y: x ** 2 + y ** 2 - 10, right=10, left=0, compare="<=", basic=plotting.change(plt)))
print(plotting.c_unary(lambda x: x ** x, start=-10, end=10, interval=100, right=2, left=-2, top=2, bottom=-2,
                       complexity=20, character="-"))

# random
print("""\033[92m
>>> print(random.gauss(0, 1, [2, 3, 4]))
>>> print(random.rand([2, 3, 4]))
>>> print(random.randint(0, 9, [2, 3, 4]))
>>> print(random.uniform(0, 9, [2, 3, 4]))
\033[m""")
print(random.gauss(0, 1, [2, 3, 4]))
print(random.rand([2, 3, 4]))
print(random.randint(0, 9, [2, 3, 4]))
print(random.uniform(0, 9, [2, 3, 4]))

# regression
print("""\033[92m
>>> print(regression.linear_regression(range(5), [2, 4, 6, 7, 8]))
>>> print(regression.parabolic_regression(range(5), [2, 4, 6, 7, 8]))
\033[m""")
print(regression.linear_regression(range(5), [2, 4, 6, 7, 8]))
print(regression.parabolic_regression(range(5), [2, 4, 6, 7, 8]))

# tools
print("""\033[92m
>>> print(tools.classify([1, 2.3, 4 + 5j, "string", list, True, 3.14, False, tuple, tools]))
>>> print(tools.deduplicate(["Python", 6, "NumPy", int, "PyPyNum", 9, "pypynum", "NumPy", 6, True]))
>>> print(tools.frange(0, 3, 0.4))
>>> print(tools.linspace(0, 2.8, 8))
\033[m""")
print(tools.classify([1, 2.3, 4 + 5j, "string", list, True, 3.14, False, tuple, tools]))
print(tools.deduplicate(["Python", 6, "NumPy", int, "PyPyNum", 9, "pypynum", "NumPy", 6, True]))
print(tools.frange(0, 3, 0.4))
print(tools.linspace(0, 2.8, 8))

for _ in list(globals().keys()):
    del globals()[_]
del _

print("""\033[91m
提示：

测试已成功通过并结束。

这些测试只是这个包功能的一部分。

更多的功能需要自己探索和尝试！

Tip:

The test has been successfully passed and ended.

These tests are only part of the functionality of this package.

More features need to be explored and tried by yourself!
\033[m""")
