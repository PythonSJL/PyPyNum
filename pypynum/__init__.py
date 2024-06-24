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
from .Array import array, fill, full, full_like, zeros, zeros_like, ones, ones_like
from . import chars
from .cipher import *
from . import constants
from .equations import *
from . import errors
from .file import read, write
from .FourierT import *
from .Geometry import *
from .Graph import *
from .Group import group
from .Logic import *
from .maths import *
from .Matrix import mat, identity, rotate90, lu, qr, hessenberg, eigen, svd
from .NeuralN import *
from .numbers import *
from .plotting import unary, binary, c_unary, color
from .polynomial import *
from .probability import *
from .Quaternion import quat, euler
from .random import *
from .regression import *
from .sequence import *
from .Symbolics import *
from .Tensor import ten, tensorproduct
from .tools import *
from .Tree import *
from . import types
from .ufuncs import *
from .utils import OrderedSet, InfIterator, LinkedList
from .Vector import vec

__version__ = "1.9.2"
print("PyPyNum", "Version -> " + __version__, "PyPI -> https://pypi.org/project/PyPyNum/",
      "Gitee -> https://www.gitee.com/PythonSJL/PyPyNum", "GitHub -> https://github.com/PythonSJL/PyPyNum", sep=" | ")
del math, arr, ite, num, real, geom, ContentError, RandomError, LogicError, InputError, FullError, Union
