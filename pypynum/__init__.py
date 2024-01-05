from .Array import array
from . import constants
from .equations import pe, mles
from . import errors
from .Geometry import *
from .Logic import *
from .mathematics import *
from .Matrix import mat, identity, lu, qr, eig, svd
from .nn import NeuralNetwork
from .plotting import unary, binary, c_unary
from .Quaternion import quat, euler
from .random import *
from .regression import *
from .Symbolics import *
from .Tensor import ten
from .tools import *
from . import types
from .Vector import vec

__version__ = "1.1.2"
__all__ = ["A", "AND", "Array", "Basic", "Binary", "C", "Circle", "FullAdder", "Geometry", "HalfAdder", "Line", "Logic",
           "Matrix", "NAND", "NOR", "NOT", "NeuralNetwork", "OR", "Point", "Polygon", "Quadrilateral", "Quaternion",
           "Symbolics", "Tensor", "Ternary", "Triangle", "Unary", "Vector", "XNOR", "XOR", "acos", "acosh", "acot",
           "acoth", "acsc", "acsch", "arrangement", "array", "asec", "asech", "asin", "asinh", "atan", "atanh", "basic",
           "beta", "binary", "c_unary", "choice", "classify", "combination", "connector", "constants", "cos", "cosh",
           "cot", "coth", "csc", "csch", "deduplicate", "definite_integral", "derivative", "distance", "eig", "english",
           "equations", "erf", "errors", "euler", "exp", "factorial", "frange", "freq", "gamma", "gauss", "gauss_error",
           "gaussian", "greek", "identity", "interpreter", "linear_regression", "linspace", "ln", "lu", "mat",
           "mathematics", "mean", "median", "mles", "mode", "nn", "operators", "parabolic_regression", "pe", "pi",
           "plotting", "product", "ptp", "qr", "quat", "rand", "randint", "random", "regression", "root", "sec", "sech",
           "sigma", "sigmoid", "sign", "sin", "sinh", "std", "svd", "tan", "tanh", "ten", "tools", "types", "unary",
           "uniform", "valid", "var", "vec", "zeta"]
print("PyPyNum", "Version -> " + __version__, "PyPI -> https://pypi.org/project/PyPyNum/",
      "Gitee -> https://www.gitee.com/PythonSJL/PyPyNum", sep=" | ")
del math, arr, ite, num, real, geom, RandomError, LogicError, InputError, FullError
