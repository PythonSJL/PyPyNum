r"""
 ________   ___    ___  ________   ___    ___  ________    ___  ___   _____ ______
|\   __  \ |\  \  /  /||\   __  \ |\  \  /  /||\   ___  \ |\  \|\  \ |\   _ \  _   \
\ \  \|\  \\ \  \/  / /\ \  \|\  \\ \  \/  / /\ \  \\ \  \\ \  \\\  \\ \  \\\__\ \  \
 \ \   ____\\ \    / /  \ \   ____\\ \    / /  \ \  \\ \  \\ \  \\\  \\ \  \\|__| \  \
  \ \  \___| \/  /  /    \ \  \___| \/  /  /    \ \  \\ \  \\ \  \\\  \\ \  \    \ \  \
   \ \__\  __/  / /       \ \__\  __/  / /       \ \__\\ \__\\ \_______\\ \__\    \ \__\
    \|__| |\___/ /         \|__| |\___/ /         \|__| \|__| \|_______| \|__|     \|__|
          \|___|/                \|___|/
"""
from .Array import array, fill, function, zeros, zeros_like
from .cipher import *
from . import constants
from .equations import *
from . import errors
from .file import read, write
from .Geometry import *
from .Group import group
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

__version__ = "1.3.1"
print("PyPyNum", "Version -> " + __version__, "PyPI -> https://pypi.org/project/PyPyNum/",
      "Gitee -> https://www.gitee.com/PythonSJL/PyPyNum", sep=" | ")
del math, arr, ite, num, real, geom, RandomError, LogicError, InputError, FullError, Union
