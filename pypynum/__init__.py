from .Array import array
from .Vector import vec
from .Matrix import identity, mat, lu, qr, eig, svd
from .Tensor import ten
from .Quaternion import quat, euler
from .Geometry import *
from .Symbolics import *
from .equations import pe, mles
from .mathematics import *
from .regression import *
from .plotting import unary, binary, c_unary
from .tools import *

Version = "1.1.0"
del math, arr, ite, num, real, geom
print("PyPyNum", "Version -> " + Version, "PyPI -> https://pypi.org/project/PyPyNum/",
      "Gitee -> https://www.gitee.com/PythonSJL/PyPyNum", sep=" | ")
del Version
