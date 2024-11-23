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
- Copyright: Copyright (c) 2023, Shen Jiayi. All rights reserved.
"""

__author__ = "Shen Jiayi"
__email__ = "2261748025@qq.com"
__copyright__ = "Copyright (c) 2023, Shen Jiayi. All rights reserved."

from . import consts
from . import zh_cn
from .arrays import *
from .chars import int2superscript, superscript2int, int2subscript, subscript2int
from .ciphers import *
from .crandom import *
from .dataproc import *
from .dists import *
from .equations import *
from .fft import *
from .files import *
from .geoms import *
from .graphs import *
from .groups import *
from .hypcmpnms import *
from .images import *
from .interp import *
from .kernels import *
from .logics import *
from .maths import *
from .matrices import *
from .multiprec import *
from .networks import *
from .numbers import *
from .plotting import *
from .polys import *
from .pprinters import *
from .random import *
from .regs import *
from .seqs import *
from .special import *
from .stattest import *
from .symbols import *
from .tensors import *
from .tools import *
from .trees import *
from .types import config
from .ufuncs import *
from .utils import *
from .vectors import *

__version__ = "1.17.0"
print("PyPyNum", "Version -> " + __version__, "PyPI -> https://pypi.org/project/PyPyNum/",
      "Gitee -> https://www.gitee.com/PythonSJL/PyPyNum", "GitHub -> https://github.com/PythonSJL/PyPyNum", sep=" | ")
for key, value in tuple(globals().items()):
    if key.endswith("Error") or str(value).startswith("typing."):
        del globals()[key]
