from .Array import array
from .cipher import *
from . import constants
from .equations import pe, mles
from . import errors
from .Geometry import *
from .Logic import *
from .mathematics import *
from .Matrix import mat, identity, lu, qr, eig, svd
from .neuralnetwork import NeuralNetwork
from .plotting import unary, binary, c_unary
from .Quaternion import quat, euler
from .random import *
from .regression import *
from .Symbolics import *
from .Tensor import ten
from .tools import *
from . import types
from .Vector import vec

__version__ = "1.2.0"
print("PyPyNum", "Version -> " + __version__, "PyPI -> https://pypi.org/project/PyPyNum/",
      "Gitee -> https://www.gitee.com/PythonSJL/PyPyNum", sep=" | ")
del math, arr, ite, num, real, geom, RandomError, LogicError, InputError, FullError
