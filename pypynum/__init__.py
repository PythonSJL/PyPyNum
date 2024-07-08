r"""
 ________   ___    ___  ________   ___    ___  ________    ___  ___   _____ ______
|\   __  \ |\  \  /  /||\   __  \ |\  \  /  /||\   ___  \ |\  \|\  \ |\   _ \  _   \
\ \  \|\  \\ \  \/  / /\ \  \|\  \\ \  \/  / /\ \  \\ \  \\ \  \\\  \\ \  \\\__\ \  \
 \ \   ____\\ \    / /  \ \   ____\\ \    / /  \ \  \\ \  \\ \  \\\  \\ \  \\|__| \  \
  \ \  \___| \/  /  /    \ \  \___| \/  /  /    \ \  \\ \  \\ \  \\\  \\ \  \    \ \  \
   \ \__\  __/  / /       \ \__\  __/  / /       \ \__\\ \__\\ \_______\\ \__\    \ \__\
    \|__| |\___/ /         \|__| |\___/ /         \|__| \|__| \|_______| \|__|     \|__|
          \|___|/                \|___|/

PyPyNum
==========

PyPyNum is a Python library for math & science computations, covering algebra, calculus, stats, with data structures
like matrices, vectors, tensors. It offers numerical tools, programs, and supports computational ops, functions,
processing, simulation, & visualization in data science & ML, crucial for research, engineering, & data processing.

Copyright
==========

- Author: Shen Jiayi
- Email: 2261748025@qq.com
- Copyright: 2023 to perpetuity. All rights reserved.
"""

__author__ = "Shen Jiayi"
__email__ = "2261748025@qq.com"
__copyright__ = "2023 to perpetuity. All rights reserved."

from .Array import array, fill, full, full_like, zeros, zeros_like, ones, ones_like, aslist, asarray
from . import chars
from .cipher import *
from . import constants
from .dists import *
from .equations import *
from . import errors
from .file import read, write
from .FourierT import *
from .Geometry import *
from .Graph import *
from .Group import group
from .highprec import *
from .image import PNG
from .Logic import *
from .maths import *
from .Matrix import mat, identity, rotate90, lu, qr, hessenberg, eigen, svd, tril_indices, rank_decomp, cholesky
from .NeuralN import *
from .numbers import *
from .plotting import unary, binary, c_unary, color
from .polynomial import *
from .pprinters import *
from .Quaternion import quat, euler
from .random import *
from .regression import *
from .sequence import *
from .stattest import *
from .Symbolics import *
from .Tensor import ten, tensorproduct
from .tools import *
from .Tree import *
from . import types
from .ufuncs import *
from .utils import OrderedSet, InfIterator, LinkedList
from .Vector import vec

__version__ = "1.11.1"
print("PyPyNum", "Version -> " + __version__, "PyPI -> https://pypi.org/project/PyPyNum/",
      "Gitee -> https://www.gitee.com/PythonSJL/PyPyNum", "GitHub -> https://github.com/PythonSJL/PyPyNum", sep=" | ")
del math, arr, ite, num, real, geom, ContentError, RandomError, LogicError, InputError, FullError, Union
