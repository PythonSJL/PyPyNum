# <font color = blue>PyPyNum</font>

<font color = gree>PyPyNum is a Python library for math & science computations, covering algebra, calculus, stats, with
data structures like matrices, vectors, tensors. It offers numerical tools, programs, and supports computational ops,
functions, processing, simulation, & visualization in data science & ML, crucial for research, engineering, & data
processing.</font><font color = red>[Python>=3.4]</font>

```
 ________   ___    ___  ________   ___    ___  ________    ___  ___   _____ ______
|\   __  \ |\  \  /  /||\   __  \ |\  \  /  /||\   ___  \ |\  \|\  \ |\   _ \  _   \
\ \  \|\  \\ \  \/  / /\ \  \|\  \\ \  \/  / /\ \  \\ \  \\ \  \\\  \\ \  \\\__\ \  \
 \ \   ____\\ \    / /  \ \   ____\\ \    / /  \ \  \\ \  \\ \  \\\  \\ \  \\|__| \  \
  \ \  \___| \/  /  /    \ \  \___| \/  /  /    \ \  \\ \  \\ \  \\\  \\ \  \    \ \  \
   \ \__\  __/  / /       \ \__\  __/  / /       \ \__\\ \__\\ \_______\\ \__\    \ \__\
    \|__| |\___/ /         \|__| |\___/ /         \|__| \|__| \|_______| \|__|     \|__|
          \|___|/                \|___|/
```

[![Downloads](https://static.pepy.tech/badge/pypynum)](https://pepy.tech/project/pypynum)
[![Downloads](https://static.pepy.tech/badge/pypynum/month)](https://pepy.tech/project/pypynum)
[![Downloads](https://static.pepy.tech/badge/pypynum/week)](https://pepy.tech/project/pypynum)

## Version -> 1.14.1 | PyPI -> https://pypi.org/project/PyPyNum/ | Gitee -> https://www.gitee.com/PythonSJL/PyPyNum | GitHub -> https://github.com/PythonSJL/PyPyNum

![LOGO](PyPyNum.png)

The logo cannot be displayed on PyPI, it can be viewed in Gitee or GitHub.

### Introduction

+ Multi functional math library, similar to numpy, scipy, etc., designed specifically for PyPy interpreters and also
  supports other types of Python interpreters
+ Update versions periodically to add more practical features
+ If you need to contact, please add QQ number 2261748025 (Py𝙿𝚢𝚝𝚑𝚘𝚗-水晶兰), or through my email 2261748025@qq.com

### Name and Function Introduction of Submodules

|    Submodule Name    |        Function Introduction        |
|:--------------------:|:-----------------------------------:|
|   `pypynum.Array`    |       Multidimensional array        |
|   `pypynum.bessel`   |          Bessel functions           |
|   `pypynum.chars`    |    Special mathematical symbols     |
|   `pypynum.cipher`   | Encryption and decryption algorithm |
|   `pypynum.confs`    |       Universal configuration       |
| `pypynum.constants`  |    Set of mathematical constants    |
|  `pypynum.crandom`   |        Random complex number        |
|  `pypynum.dataproc`  |           Data processing           |
|   `pypynum.dists`    |      Probability distribution       |
| `pypynum.equations`  |          Solving equations          |
|   `pypynum.errors`   |          Exception object           |
|    `pypynum.file`    |         File read and write         |
|  `pypynum.FourierT`  |          Fourier transform          |
|  `pypynum.Geometry`  |           Geometric shape           |
|   `pypynum.Graph`    |       Graph Theory Algorithm        |
|   `pypynum.Group`    |       Group Theory Algorithm        |
|   `pypynum.image`    |          Image processing           |
|   `pypynum.interp`   |         Data Interpolation          |
|   `pypynum.Logic`    |        Logic circuit design         |
|   `pypynum.maths`    |   General mathematical functions    |
|   `pypynum.Matrix`   |          Matrix operation           |
| `pypynum.multiprec`  |     Multi precision calculation     |
|  `pypynum.NeuralN`   |       Neural network training       |
|  `pypynum.numbers`   |          Number processing          |
|  `pypynum.plotting`  |         Data visualization          |
| `pypynum.polynomial` |        Polynomial operation         |
| `pypynum.pprinters`  |           Pretty printers           |
| `pypynum.Quaternion` |        Quaternion operation         |
|   `pypynum.random`   |      Random number generation       |
| `pypynum.regression` |         Regression analysis         |
|  `pypynum.sequence`  |        Sequence calculation         |
|  `pypynum.stattest`  |          Statistical test           |
| `pypynum.Symbolics`  |         Symbol calculation          |
|   `pypynum.Tensor`   |          Tensor operation           |
|    `pypynum.test`    |              Easy test              |
|    `pypynum.this`    |           Zen of Projects           |
|   `pypynum.tools`    |         Auxiliary functions         |
|    `pypynum.Tree`    |         Tree data structure         |
|   `pypynum.types`    |            Special types            |
|   `pypynum.ufuncs`   |         Universal functions         |
|   `pypynum.utils`    |               Utility               |
|   `pypynum.Vector`   |          Vector operation           |
|   `pypynum.zh_cn`    |    Functions with Chinese names     |

### The Zen of PyPyNum (Preview)

```
                The Zen of PyPyNum, by Shen Jiayi

In this mathematical sanctuary, we weave our algorithms with pure Python threads.
Precision outweighs approximation.
Elegance in mathematics transcends the bulky algorithms.
Clarity in logic illuminates the darkest problems.
Simplicity in form is the pinnacle of sophistication.
Flat hierarchies in our code mirror the linear nature of functions.
Sparse code, like a minimal polynomial, retains essence without redundancy.
```

```
...

Do you want to view all the content?

Enter "from pypynum import this" in your

Python interpreter and run it!
```

```
                                                                September 5, 2024
```

### Functional changes compared to the previous version

```
!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=

< Fixed the display issue of multi precision complex numbers >

!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=

< Updated the Zen content of the project >

!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=

< The following two function names have removed the 'generate' prefix >

generate_primes(limit: int) -> list
generate_semiprimes(limit: int) -> list

!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=

<<< Newly added functions and classes >>>

PyPyNum
├── Matrix
│   └── FUNCTION
│       └── triu_indices(n: int, k: int, m: int) -> tuple  # Returns the indices of the upper triangular part of a matrix
├── multiprec
│   └── FUNCTION
│       ├── mp_euler_gamma(sigfigs: int) -> decimal.Decimal  # Returns the Euler-Mascheroni constant with specified significant figures
├── sequence
│   └── FUNCTION
│       ├── bell_triangle(n: int) -> list  # Returns the first n rows of the Bell triangle
│       ├── pascal_triangle(n: int) -> list  # Returns the first n rows of Pascal's triangle
│       ├── stirling1(n: int) -> list  # Returns the first n rows of Stirling numbers of the first kind
│       └── stirling2(n: int) -> list  # Returns the first n rows of Stirling numbers of the second kind
├── tools
│   └── FUNCTION
│       └── twinprimes(limit: int) -> list  # Returns all twin prime pairs up to and including the specified limit

!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=
```

### Run Time Test

Python interpreter version

+ CPython 3.8.10

+ PyPy 3.10.12

| Matrix Time Test                               | NumPy+CPython (seconds)                                                                                                                                                            | Ranking | PyPyNum+PyPy (seconds)                                                                                                                                                             | Ranking | Mpmath_+_PyPy_ (seconds)                                                                                                                                                         | Ranking | SymPy_+_PyPy_ (seconds)                                                                                                                                                                                                                    | Ranking |
|------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| Create a hundred order random number matrix    | 0.000083                                                                                                                                                                           | 1       | 0.005374                                                                                                                                                                           | 2       | 0.075253                                                                                                                                                                         | 3       | 0.230530                                                                                                                                                                                                                                   | 4       |
| Create a thousand order random number matrix   | 0.006740                                                                                                                                                                           | 1       | 0.035666                                                                                                                                                                           | 2       | 1.200950                                                                                                                                                                         | 3       | 4.370265                                                                                                                                                                                                                                   | 4       |
| Addition of matrices of order one hundred      | 0.000029                                                                                                                                                                           | 1       | 0.002163                                                                                                                                                                           | 2       | 0.045641                                                                                                                                                                         | 4       | 0.035700                                                                                                                                                                                                                                   | 3       |
| Adding matrices of order one thousand          | 0.002647                                                                                                                                                                           | 1       | 0.019111                                                                                                                                                                           | 2       | 1.746957                                                                                                                                                                         | 4       | 0.771542                                                                                                                                                                                                                                   | 3       |
| Determinant of a hundred order matrix          | 0.087209                                                                                                                                                                           | 2       | 0.016331                                                                                                                                                                           | 1       | 4.354507                                                                                                                                                                         | 3       | 5.157206                                                                                                                                                                                                                                   | 4       |
| Determinant of a thousand order matrix         | 0.616113                                                                                                                                                                           | 1       | 3.509747                                                                                                                                                                           | 2       | It takes a long time                                                                                                                                                             | 3       | It takes a long time                                                                                                                                                                                                                       | 4       |
| Finding the inverse of a hundred order matrix  | 0.162770                                                                                                                                                                           | 2       | 0.015768                                                                                                                                                                           | 1       | 8.162948                                                                                                                                                                         | 3       | 21.437424                                                                                                                                                                                                                                  | 4       |
| Finding the inverse of a thousand order matrix | 0.598905                                                                                                                                                                           | 1       | 17.072552                                                                                                                                                                          | 2       | It takes a long time                                                                                                                                                             | 3       | It takes a long time                                                                                                                                                                                                                       | 4       |
| Array output effect                            | ```[[[[ -7 -67]```<br>```[-78  29]]```<br><br>```[[-86 -97]```<br>```[ 68  -3]]]```<br><br><br>```[[[ 11  42]```<br>```[ 24 -65]]```<br><br>```[[-60  72]```<br>```[ 73   2]]]]``` | /       | ```[[[[ 37  83]```<br>```[ 40   2]]```<br><br>```[[ -5 -34]```<br>```[ -7  72]]]```<br><br><br>```[[[ 13 -64]```<br>```[  6  90]]```<br><br>```[[ 68  57]```<br>```[ 78  11]]]]``` | /       | ```[-80.0   -8.0  80.0  -88.0]```<br>```[-99.0  -43.0  87.0   81.0]```<br>```[ 20.0  -55.0  98.0    8.0]```<br>```[  8.0   44.0  64.0  -35.0]```<br><br>(Only supports matrices) | /       | ```⎡⎡16   -56⎤  ⎡ 8   -28⎤⎤```<br>```⎢⎢        ⎥  ⎢        ⎥⎥```<br>```⎢⎣-56  56 ⎦  ⎣-28  28 ⎦⎥```<br>```⎢                      ⎥```<br>```⎢ ⎡-2  7 ⎤   ⎡-18  63 ⎤⎥```<br>```⎢ ⎢      ⎥   ⎢        ⎥⎥```<br>```⎣ ⎣7   -7⎦   ⎣63   -63⎦⎦``` | /       |

### Basic structure

```
PyPyNum
├── Array
│   ├── CLASS
│   │   ├── Array(object)/__init__(self: Any, data: Any, check: Any) -> Any
│   │   └── BoolArray(pypynum.Array.Array)/__init__(self: Any, data: Any, check: Any) -> Any
│   └── FUNCTION
│       ├── array(data: Any) -> Any
│       ├── asarray(data: Any) -> Any
│       ├── aslist(data: Any) -> Any
│       ├── boolarray(data: Any) -> Any
│       ├── fill(shape: Any, sequence: Any, repeat: Any, pad: Any, rtype: Any) -> Any
│       ├── full(shape: Any, fill_value: Any, rtype: Any) -> Any
│       ├── full_like(a: Any, fill_value: Any, rtype: Any) -> Any
│       ├── get_shape(data: Any) -> Any
│       ├── is_valid_array(_array: Any, _shape: Any) -> Any
│       ├── ones(shape: Any, rtype: Any) -> Any
│       ├── ones_like(a: Any, rtype: Any) -> Any
│       ├── zeros(shape: Any, rtype: Any) -> Any
│       └── zeros_like(a: Any, rtype: Any) -> Any
├── FourierT
│   ├── CLASS
│   │   └── FT1D(object)/__init__(self: Any, data: Any) -> Any
│   └── FUNCTION
├── Geometry
│   ├── CLASS
│   │   ├── Circle(object)/__init__(self: Any, center: typing.Union[list, tuple], radius: typing.Union[int, float]) -> Any
│   │   ├── Line(object)/__init__(self: Any, a: typing.Union[list, tuple], b: typing.Union[list, tuple]) -> Any
│   │   ├── Point(object)/__init__(self: Any, p: typing.Union[list, tuple]) -> Any
│   │   ├── Polygon(object)/__init__(self: Any, p: typing.Union[list, tuple]) -> Any
│   │   ├── Quadrilateral(object)/__init__(self: Any, a: typing.Union[list, tuple], b: typing.Union[list, tuple], c: typing.Union[list, tuple], d: typing.Union[list, tuple]) -> Any
│   │   └── Triangle(object)/__init__(self: Any, a: typing.Union[list, tuple], b: typing.Union[list, tuple], c: typing.Union[list, tuple]) -> Any
│   └── FUNCTION
│       └── distance(g1: Any, g2: Any, error: typing.Union[int, float]) -> float
├── Graph
│   ├── CLASS
│   │   ├── BaseGraph(object)/__init__(self: Any) -> Any
│   │   ├── BaseWeGraph(pypynum.Graph.BaseGraph)/__init__(self: Any) -> Any
│   │   ├── DiGraph(pypynum.Graph.BaseGraph)/__init__(self: Any) -> Any
│   │   ├── UnGraph(pypynum.Graph.BaseGraph)/__init__(self: Any) -> Any
│   │   ├── WeDiGraph(pypynum.Graph.BaseWeGraph)/__init__(self: Any) -> Any
│   │   └── WeUnGraph(pypynum.Graph.BaseWeGraph)/__init__(self: Any) -> Any
│   └── FUNCTION
├── Group
│   ├── CLASS
│   │   └── Group(object)/__init__(self: Any, data: Any, operation: Any) -> Any
│   └── FUNCTION
│       └── group(data: Any) -> Any
├── Logic
│   ├── CLASS
│   │   ├── AND(pypynum.Logic.Binary)/__init__(self: Any, label: Any, pin0: Any, pin1: Any) -> Any
│   │   ├── Basic(object)/__init__(self: Any, label: Any) -> Any
│   │   ├── Binary(pypynum.Logic.Basic)/__init__(self: Any, label: Any, pin0: Any, pin1: Any) -> Any
│   │   ├── COMP(pypynum.Logic.Binary)/__init__(self: Any, label: Any, pin0: Any, pin1: Any) -> Any
│   │   ├── DFF(pypynum.Logic.Unary)/__init__(self: Any, label: Any, pin0: Any, state: Any) -> Any
│   │   ├── FullAdder(pypynum.Logic.Ternary)/__init__(self: Any, label: Any, pin0: Any, pin1: Any, pin2: Any) -> Any
│   │   ├── FullSuber(pypynum.Logic.Ternary)/__init__(self: Any, label: Any, pin0: Any, pin1: Any, pin2: Any) -> Any
│   │   ├── HalfAdder(pypynum.Logic.Binary)/__init__(self: Any, label: Any, pin0: Any, pin1: Any) -> Any
│   │   ├── HalfSuber(pypynum.Logic.Binary)/__init__(self: Any, label: Any, pin0: Any, pin1: Any) -> Any
│   │   ├── JKFF(pypynum.Logic.Binary)/__init__(self: Any, label: Any, pin0: Any, pin1: Any, state: Any) -> Any
│   │   ├── NAND(pypynum.Logic.Binary)/__init__(self: Any, label: Any, pin0: Any, pin1: Any) -> Any
│   │   ├── NOR(pypynum.Logic.Binary)/__init__(self: Any, label: Any, pin0: Any, pin1: Any) -> Any
│   │   ├── NOT(pypynum.Logic.Unary)/__init__(self: Any, label: Any, pin0: Any) -> Any
│   │   ├── OR(pypynum.Logic.Binary)/__init__(self: Any, label: Any, pin0: Any, pin1: Any) -> Any
│   │   ├── Quaternary(pypynum.Logic.Basic)/__init__(self: Any, label: Any, pin0: Any, pin1: Any, pin2: Any, pin3: Any) -> Any
│   │   ├── TFF(pypynum.Logic.Unary)/__init__(self: Any, label: Any, pin0: Any, state: Any) -> Any
│   │   ├── Ternary(pypynum.Logic.Basic)/__init__(self: Any, label: Any, pin0: Any, pin1: Any, pin2: Any) -> Any
│   │   ├── TwoBDiver(pypynum.Logic.Quaternary)/__init__(self: Any, label: Any, pin0: Any, pin1: Any, pin2: Any, pin3: Any) -> Any
│   │   ├── TwoBMuler(pypynum.Logic.Quaternary)/__init__(self: Any, label: Any, pin0: Any, pin1: Any, pin2: Any, pin3: Any) -> Any
│   │   ├── Unary(pypynum.Logic.Basic)/__init__(self: Any, label: Any, pin0: Any) -> Any
│   │   ├── XNOR(pypynum.Logic.Binary)/__init__(self: Any, label: Any, pin0: Any, pin1: Any) -> Any
│   │   └── XOR(pypynum.Logic.Binary)/__init__(self: Any, label: Any, pin0: Any, pin1: Any) -> Any
│   └── FUNCTION
│       └── connector(previous: Any, latter: Any) -> Any
├── Matrix
│   ├── CLASS
│   │   └── Matrix(pypynum.Array.Array)/__init__(self: Any, data: Any, check: Any) -> Any
│   └── FUNCTION
│       ├── cholesky(matrix: Any, hermitian: Any) -> Any
│       ├── eigen(matrix: pypynum.Matrix.Matrix) -> tuple
│       ├── hessenberg(matrix: pypynum.Matrix.Matrix) -> tuple
│       ├── identity(n: int) -> pypynum.Matrix.Matrix
│       ├── lu(matrix: pypynum.Matrix.Matrix) -> tuple
│       ├── mat(data: Any) -> Any
│       ├── qr(matrix: pypynum.Matrix.Matrix) -> tuple
│       ├── rank_decomp(matrix: pypynum.Matrix.Matrix) -> tuple
│       ├── rotate90(matrix: pypynum.Matrix.Matrix, times: int) -> pypynum.Matrix.Matrix
│       ├── svd(matrix: pypynum.Matrix.Matrix) -> tuple
│       ├── tril_indices(n: int, k: int, m: int) -> tuple
│       └── triu_indices(n: int, k: int, m: int) -> tuple
├── NeuralN
│   ├── CLASS
│   │   └── NeuralNetwork(object)/__init__(self: Any, _input: Any, _hidden: Any, _output: Any) -> Any
│   └── FUNCTION
│       └── neuraln(_input: Any, _hidden: Any, _output: Any) -> Any
├── Quaternion
│   ├── CLASS
│   │   ├── Euler(object)/__init__(self: Any, y: typing.Union[int, float], p: typing.Union[int, float], r: typing.Union[int, float]) -> Any
│   │   └── Quaternion(object)/__init__(self: Any, w: typing.Union[int, float], x: typing.Union[int, float], y: typing.Union[int, float], z: typing.Union[int, float]) -> Any
│   └── FUNCTION
│       ├── change(data: typing.Union[pypynum.Quaternion.Quaternion, pypynum.Matrix.Matrix, pypynum.Quaternion.Euler], to: str) -> typing.Union[pypynum.Quaternion.Quaternion, pypynum.Matrix.Matrix, pypynum.Quaternion.Euler]
│       ├── euler(yaw: typing.Union[int, float], pitch: typing.Union[int, float], roll: typing.Union[int, float]) -> pypynum.Quaternion.Euler
│       └── quat(w: typing.Union[int, float], x: typing.Union[int, float], y: typing.Union[int, float], z: typing.Union[int, float]) -> pypynum.Quaternion.Quaternion
├── Symbolics
│   ├── CLASS
│   └── FUNCTION
│       └── parse_expr(expr: str) -> list
├── Tensor
│   ├── CLASS
│   │   └── Tensor(pypynum.Array.Array)/__init__(self: Any, data: Any, check: Any) -> Any
│   └── FUNCTION
│       ├── ten(data: list) -> pypynum.Tensor.Tensor
│       ├── tensor_and_number(tensor: Any, operator: Any, number: Any) -> Any
│       ├── tensorproduct(tensors: pypynum.Tensor.Tensor) -> pypynum.Tensor.Tensor
│       ├── zeros(_dimensions: Any) -> Any
│       └── zeros_like(_nested_list: Any) -> Any
├── Tree
│   ├── CLASS
│   │   ├── MultiTree(object)/__init__(self: Any, root: Any) -> Any
│   │   └── MultiTreeNode(object)/__init__(self: Any, data: Any) -> Any
│   └── FUNCTION
├── Vector
│   ├── CLASS
│   │   └── Vector(pypynum.Array.Array)/__init__(self: Any, data: Any, check: Any) -> Any
│   └── FUNCTION
│       └── vec(data: Any) -> Any
├── bessel
│   ├── CLASS
│   └── FUNCTION
│       ├── bessel_i0(x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
│       ├── bessel_i1(x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
│       ├── bessel_iv(v: typing.Union[int, float], x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
│       ├── bessel_j0(x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
│       ├── bessel_j1(x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
│       └── bessel_jv(v: typing.Union[int, float], x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
├── chars
│   ├── CLASS
│   └── FUNCTION
│       ├── int2subscript(standard_str: str) -> str
│       ├── int2superscript(standard_str: str) -> str
│       ├── subscript2int(subscript_str: str) -> str
│       └── superscript2int(superscript_str: str) -> str
├── cipher
│   ├── CLASS
│   └── FUNCTION
│       ├── atbash(text: str) -> str
│       ├── base_64(text: str, decrypt: bool) -> str
│       ├── caesar(text: str, shift: int, decrypt: bool) -> str
│       ├── hill256(text: bytes, key: list, decrypt: bool) -> bytes
│       ├── ksa(key: bytes) -> list
│       ├── morse(text: str, decrypt: bool) -> str
│       ├── playfair(text: str, key: str, decrypt: bool) -> str
│       ├── prga(s: list) -> Any
│       ├── rc4(text: bytes, key: bytes) -> bytes
│       ├── rot13(text: str) -> str
│       ├── substitution(text: str, sub_map: dict, decrypt: bool) -> str
│       └── vigenere(text: str, key: str, decrypt: bool) -> str
├── confs
│   ├── CLASS
│   └── FUNCTION
├── constants
│   ├── CLASS
│   └── FUNCTION
├── crandom
│   ├── CLASS
│   └── FUNCTION
│       ├── randint_polar(left: int, right: int, mod: typing.Union[int, float], angle: typing.Union[int, float]) -> complex
│       ├── randint_rect(left: int, right: int, real: typing.Union[int, float], imag: typing.Union[int, float]) -> complex
│       ├── random_polar(mod: typing.Union[int, float], angle: typing.Union[int, float]) -> complex
│       ├── random_rect(real: typing.Union[int, float], imag: typing.Union[int, float]) -> complex
│       ├── uniform_polar(left: typing.Union[int, float], right: typing.Union[int, float], mod: typing.Union[int, float], angle: typing.Union[int, float]) -> complex
│       └── uniform_rect(left: typing.Union[int, float], right: typing.Union[int, float], real: typing.Union[int, float], imag: typing.Union[int, float]) -> complex
├── dataproc
│   ├── CLASS
│   │   └── Series(object)/__init__(self: Any, data: typing.Any, index: typing.Any) -> None
│   └── FUNCTION
├── dists
│   ├── CLASS
│   └── FUNCTION
│       ├── beta_pdf(x: Any, a: Any, b: Any) -> Any
│       ├── binom_pmf(k: Any, n: Any, p: Any) -> Any
│       ├── cauchy_cdf(x: Any, x0: Any, gamma: Any) -> Any
│       ├── cauchy_pdf(x: Any, x0: Any, gamma: Any) -> Any
│       ├── chi2_cdf(x: Any, df: Any) -> Any
│       ├── chi2_pdf(x: Any, df: Any) -> Any
│       ├── expon_cdf(x: Any, scale: Any) -> Any
│       ├── expon_pdf(x: Any, scale: Any) -> Any
│       ├── f_pdf(x: Any, dfnum: Any, dfden: Any) -> Any
│       ├── gamma_pdf(x: Any, shape: Any, scale: Any) -> Any
│       ├── geometric_pmf(k: Any, p: Any) -> Any
│       ├── hypergeom_pmf(k: Any, mg: Any, n: Any, nt: Any) -> Any
│       ├── inv_gauss_pdf(x: Any, mu: Any, lambda_: Any, alpha: Any) -> Any
│       ├── levy_pdf(x: Any, c: Any) -> Any
│       ├── log_logistic_cdf(x: Any, alpha: Any, beta: Any) -> Any
│       ├── log_logistic_pdf(x: Any, alpha: Any, beta: Any) -> Any
│       ├── logistic_cdf(x: Any, mu: Any, s: Any) -> Any
│       ├── logistic_pdf(x: Any, mu: Any, s: Any) -> Any
│       ├── lognorm_cdf(x: Any, mu: Any, sigma: Any) -> Any
│       ├── lognorm_pdf(x: Any, s: Any, scale: Any) -> Any
│       ├── logser_pmf(k: Any, p: Any) -> Any
│       ├── multinomial_pmf(k: Any, n: Any, p: Any) -> Any
│       ├── nbinom_pmf(k: Any, n: Any, p: Any) -> Any
│       ├── nhypergeom_pmf(k: Any, m: Any, n: Any, r: Any) -> Any
│       ├── normal_cdf(x: Any, mu: Any, sigma: Any) -> Any
│       ├── normal_pdf(x: Any, mu: Any, sigma: Any) -> Any
│       ├── pareto_pdf(x: Any, k: Any, m: Any) -> Any
│       ├── poisson_pmf(k: Any, mu: Any) -> Any
│       ├── rayleigh_pdf(x: Any, sigma: Any) -> Any
│       ├── t_pdf(x: Any, df: Any) -> Any
│       ├── uniform_cdf(x: Any, loc: Any, scale: Any) -> Any
│       ├── uniform_pdf(x: Any, loc: Any, scale: Any) -> Any
│       ├── vonmises_pdf(x: Any, mu: Any, kappa: Any) -> Any
│       ├── weibull_max_pdf(x: Any, c: Any, scale: Any, loc: Any) -> Any
│       ├── weibull_min_pdf(x: Any, c: Any, scale: Any, loc: Any) -> Any
│       └── zipf_pmf(k: Any, s: Any, n: Any) -> Any
├── equations
│   ├── CLASS
│   └── FUNCTION
│       ├── lin_eq(left: list, right: list) -> list
│       └── poly_eq(coefficients: list) -> list
├── errors
│   ├── CLASS
│   └── FUNCTION
├── file
│   ├── CLASS
│   └── FUNCTION
│       ├── read(file: str) -> list
│       └── write(file: str, cls: object) -> Any
├── image
│   ├── CLASS
│   │   └── PNG(object)/__init__(self: Any) -> None
│   └── FUNCTION
│       └── crc(data: Any, length: Any, init: Any, xor: Any) -> Any
├── interp
│   ├── CLASS
│   └── FUNCTION
│       ├── bicubic(x: Any) -> Any
│       ├── contribute(src: Any, x: Any, y: Any, channels: Any) -> Any
│       ├── interp1d(data: typing.Union[list, tuple], length: int) -> list
│       └── interp2d(src: Any, new_height: Any, new_width: Any, channels: Any, round_res: Any, min_val: Any, max_val: Any) -> Any
├── maths
│   ├── CLASS
│   └── FUNCTION
│       ├── arrangement(n: int, r: int) -> int
│       ├── combination(n: int, r: int) -> int
│       ├── acos(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── acosh(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── acot(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── acoth(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── acsc(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── acsch(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── arrangement(n: int, r: int) -> int
│       ├── asec(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── asech(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── asin(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── asinh(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── atan(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── atanh(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── average(data: typing.Union[list, tuple], weights: typing.Union[list, tuple]) -> float
│       ├── beta(p: typing.Union[int, float], q: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── central_moment(data: typing.Union[list, tuple], order: int) -> float
│       ├── coeff_det(x: typing.Union[list, tuple], y: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── combination(n: int, r: int) -> int
│       ├── corr_coeff(x: typing.Union[list, tuple], y: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── cos(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── cosh(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── cot(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── coth(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── cov(x: typing.Union[list, tuple], y: typing.Union[list, tuple], dof: int) -> typing.Union[int, float, complex]
│       ├── crt(n: typing.Union[list, tuple], a: typing.Union[list, tuple]) -> int
│       ├── csc(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── csch(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── cumprod(lst: typing.Union[list, tuple]) -> list
│       ├── cumsum(lst: typing.Union[list, tuple]) -> list
│       ├── deriv(f: Any, x: float, h: float, method: str, args: Any, kwargs: Any) -> Any
│       ├── erf(x: typing.Union[int, float]) -> float
│       ├── exgcd(a: int, b: int) -> tuple
│       ├── exp(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── factorial(n: int) -> int
│       ├── freq(data: typing.Union[list, tuple]) -> dict
│       ├── gamma(alpha: typing.Union[int, float]) -> float
│       ├── gcd(args: int) -> int
│       ├── geom_mean(numbers: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── harm_mean(numbers: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── integ(f: Any, x_start: typing.Union[int, float], x_end: typing.Union[int, float], n: int, args: Any, kwargs: Any) -> float
│       ├── iroot(y: int, n: int) -> int
│       ├── is_possibly_square(n: int) -> bool
│       ├── is_square(n: int) -> bool
│       ├── isqrt(x: int) -> int
│       ├── kurt(data: typing.Union[list, tuple], fisher: bool) -> float
│       ├── lcm(args: int) -> int
│       ├── ln(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── lower_gamma(s: typing.Union[int, float, complex], x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
│       ├── mean(numbers: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── median(numbers: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── mod_order(a: int, n: int, b: int) -> int
│       ├── mode(data: typing.Union[list, tuple]) -> Any
│       ├── normalize(data: typing.Union[list, tuple], target: typing.Union[int, float, complex]) -> typing.Union[list, tuple]
│       ├── parity(x: int) -> int
│       ├── pi(i: int, n: int, f: Any) -> typing.Union[int, float, complex]
│       ├── primitive_root(a: int, single: bool) -> typing.Union[int, list]
│       ├── product(numbers: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── ptp(numbers: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── quantile(data: list, q: float, interpolation: str, ordered: bool) -> float
│       ├── raw_moment(data: typing.Union[list, tuple], order: int) -> float
│       ├── roll(seq: typing.Union[list, tuple, str], shift: int) -> typing.Union[list, tuple, str]
│       ├── root(x: typing.Union[int, float, complex], y: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
│       ├── sec(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── sech(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── sigma(i: int, n: int, f: Any) -> typing.Union[int, float, complex]
│       ├── sigmoid(x: typing.Union[int, float]) -> float
│       ├── sign(x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
│       ├── sin(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── sinh(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── skew(data: typing.Union[list, tuple]) -> float
│       ├── square_mean(numbers: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── std(numbers: typing.Union[list, tuple], dof: int) -> typing.Union[int, float, complex]
│       ├── sumprod(arrays: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── tan(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── tanh(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── totient(n: int) -> int
│       ├── upper_gamma(s: typing.Union[int, float, complex], x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
│       ├── var(numbers: typing.Union[list, tuple], dof: int) -> typing.Union[int, float, complex]
│       ├── xlogy(x: typing.Union[int, float, complex], y: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
│       └── zeta(alpha: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
├── multiprec
│   ├── CLASS
│   │   └── MPComplex(object)/__init__(self: Any, real: Any, imag: Any, sigfigs: Any) -> Any
│   └── FUNCTION
│       ├── _remove_trailing_zeros(value: typing.Any) -> str
│       ├── _setprec(sigfigs: int) -> Any
│       ├── asmpc(real: typing.Union[int, float, str, decimal.Decimal, complex, pypynum.multiprec.MPComplex], imag: typing.Union[int, float, str, decimal.Decimal], sigfigs: int) -> pypynum.multiprec.MPComplex
│       ├── frac2dec(frac: fractions.Fraction, sigfigs: int) -> decimal.Decimal
│       ├── mp_acos(x: typing.Union[int, float, str, decimal.Decimal], sigfigs: int) -> decimal.Decimal
│       ├── mp_asin(x: typing.Union[int, float, str, decimal.Decimal], sigfigs: int) -> decimal.Decimal
│       ├── mp_atan(x: typing.Union[int, float, str, decimal.Decimal], sigfigs: int) -> decimal.Decimal
│       ├── mp_atan2(y: typing.Union[int, float, str, decimal.Decimal], x: typing.Union[int, float, str, decimal.Decimal], sigfigs: int) -> decimal.Decimal
│       ├── mp_cos(x: typing.Union[int, float, str, decimal.Decimal], sigfigs: int) -> decimal.Decimal
│       ├── mp_cosh(x: typing.Union[int, float, str, decimal.Decimal], sigfigs: int) -> decimal.Decimal
│       ├── mp_e(sigfigs: int, method: str) -> decimal.Decimal
│       ├── mp_euler_gamma(sigfigs: int) -> decimal.Decimal
│       ├── mp_exp(x: typing.Union[int, float, str, decimal.Decimal], sigfigs: int, builtin: bool) -> decimal.Decimal
│       ├── mp_fresnel_c(x: typing.Union[int, float, str, decimal.Decimal], sigfigs: int) -> decimal.Decimal
│       ├── mp_fresnel_s(x: typing.Union[int, float, str, decimal.Decimal], sigfigs: int) -> decimal.Decimal
│       ├── mp_ln(x: typing.Union[int, float, str, decimal.Decimal], sigfigs: int, builtin: bool) -> decimal.Decimal
│       ├── mp_log(x: typing.Union[int, float, str, decimal.Decimal], base: typing.Union[int, float, str, decimal.Decimal], sigfigs: int, builtin: bool) -> decimal.Decimal
│       ├── mp_phi(sigfigs: int, method: str) -> decimal.Decimal
│       ├── mp_pi(sigfigs: int, method: str) -> decimal.Decimal
│       ├── mp_sin(x: typing.Union[int, float, str, decimal.Decimal], sigfigs: int) -> decimal.Decimal
│       └── mp_sinh(x: typing.Union[int, float, str, decimal.Decimal], sigfigs: int) -> decimal.Decimal
├── numbers
│   ├── CLASS
│   └── FUNCTION
│       ├── float2fraction(number: float, mixed: bool, error: float) -> tuple
│       ├── int2roman(integer: int, overline: bool) -> str
│       ├── int2words(integer: int) -> str
│       ├── parse_float(s: str) -> tuple
│       ├── roman2int(roman_num: str) -> int
│       ├── split_float(s: str) -> tuple
│       └── str2int(string: str) -> int
├── plotting
│   ├── CLASS
│   └── FUNCTION
│       ├── background(right: typing.Union[int, float], left: typing.Union[int, float], top: typing.Union[int, float], bottom: typing.Union[int, float], complexity: typing.Union[int, float], ratio: typing.Union[int, float], string: bool) -> typing.Union[list, str]
│       ├── binary(function: Any, right: typing.Union[int, float], left: typing.Union[int, float], top: typing.Union[int, float], bottom: typing.Union[int, float], complexity: typing.Union[int, float], ratio: typing.Union[int, float], error: Any, compare: Any, string: bool, basic: list, character: str, data: bool, coloration: Any) -> typing.Union[list, str]
│       ├── c_unary(function: Any, projection: str, right: typing.Union[int, float], left: typing.Union[int, float], top: typing.Union[int, float], bottom: typing.Union[int, float], complexity: typing.Union[int, float], ratio: typing.Union[int, float], string: bool, basic: list, character: str, data: bool, coloration: Any) -> typing.Union[list, str]
│       ├── change(data: typing.Union[list, str]) -> typing.Union[list, str]
│       ├── color(text: str, rgb: typing.Union[list, tuple]) -> str
│       └── unary(function: Any, right: typing.Union[int, float], left: typing.Union[int, float], top: typing.Union[int, float], bottom: typing.Union[int, float], complexity: typing.Union[int, float], ratio: typing.Union[int, float], string: bool, basic: list, character: str, data: bool, coloration: Any) -> typing.Union[list, str]
├── polynomial
│   ├── CLASS
│   │   └── Polynomial(object)/__init__(self: Any, terms: Any) -> Any
│   └── FUNCTION
│       ├── chebgauss(n: Any) -> Any
│       ├── chebpoly(n: Any, single: Any) -> Any
│       ├── from_coeffs(coeffs: Any) -> Any
│       ├── from_coords(coords: Any) -> Any
│       ├── laggauss(n: Any) -> Any
│       ├── lagpoly(n: Any, single: Any) -> Any
│       ├── leggauss(n: Any) -> Any
│       ├── legpoly(n: Any, single: Any) -> Any
│       └── poly(terms: Any) -> Any
├── pprinters
│   ├── CLASS
│   └── FUNCTION
│       └── pprint_matrix(matrix: Any, style: Any, output: Any) -> Any
├── random
│   ├── CLASS
│   └── FUNCTION
│       ├── __create_nested_list(dimensions: Any, func: Any) -> Any
│       ├── __validate_shape(shape: Any) -> Any
│       ├── choice(seq: typing.Union[list, tuple, str], shape: typing.Union[list, tuple]) -> Any
│       ├── gauss(mu: typing.Union[int, float], sigma: typing.Union[int, float], shape: typing.Union[list, tuple]) -> typing.Union[float, list]
│       ├── rand(shape: typing.Union[list, tuple]) -> typing.Union[float, list]
│       ├── randint(a: int, b: int, shape: typing.Union[list, tuple]) -> typing.Union[int, list]
│       └── uniform(a: typing.Union[int, float], b: typing.Union[int, float], shape: typing.Union[list, tuple]) -> typing.Union[float, list]
├── regression
│   ├── CLASS
│   └── FUNCTION
│       ├── lin_reg(x: typing.Union[list, tuple], y: typing.Union[list, tuple]) -> list
│       ├── par_reg(x: typing.Union[list, tuple], y: typing.Union[list, tuple]) -> list
│       └── poly_reg(x: typing.Union[list, tuple], y: typing.Union[list, tuple], n: int) -> list
├── sequence
│   ├── CLASS
│   └── FUNCTION
│       ├── arithmetic_sequence(a1: typing.Union[int, float], an: typing.Union[int, float], d: typing.Union[int, float], n: typing.Union[int, float], s: typing.Union[int, float]) -> dict
│       ├── bell_triangle(n: int) -> list
│       ├── bernoulli(n: int, single: bool) -> list
│       ├── catalan(n: int, single: bool) -> typing.Union[int, list]
│       ├── farey(n: int) -> list
│       ├── fibonacci(n: int, single: bool) -> typing.Union[int, list]
│       ├── geometric_sequence(a1: typing.Union[int, float], an: typing.Union[int, float], r: typing.Union[int, float], n: typing.Union[int, float], s: typing.Union[int, float]) -> dict
│       ├── pascal_triangle(n: int) -> list
│       ├── recaman(n: int, single: bool) -> typing.Union[int, list]
│       ├── stirling1(n: int) -> list
│       └── stirling2(n: int) -> list
├── stattest
│   ├── CLASS
│   └── FUNCTION
│       ├── chi2_cont(contingency: list, lambda_: float, calc_p: bool, corr: bool) -> tuple
│       ├── chisquare(observed: list, expected: list) -> tuple
│       ├── kurttest(data: list, two_tailed: bool) -> tuple
│       ├── mediantest(samples: Any, ties: Any, lambda_: Any, corr: Any) -> Any
│       ├── normaltest(data: list) -> tuple
│       └── skewtest(data: list, two_tailed: bool) -> tuple
├── test
│   ├── CLASS
│   └── FUNCTION
├── this
│   ├── CLASS
│   └── FUNCTION
├── tools
│   ├── CLASS
│   └── FUNCTION
│       ├── classify(array: typing.Union[list, tuple]) -> dict
│       ├── dedup(iterable: typing.Union[list, tuple, str]) -> typing.Union[list, tuple, str]
│       ├── frange(start: typing.Union[int, float], stop: typing.Union[int, float], step: float) -> list
│       ├── geomspace(start: typing.Union[int, float], stop: typing.Union[int, float], number: int) -> list
│       ├── levenshtein_distance(s1: str, s2: str) -> int
│       ├── linspace(start: typing.Union[int, float], stop: typing.Union[int, float], number: int) -> list
│       ├── magic_square(n: Any) -> Any
│       ├── primality(n: int, iter_num: int) -> bool
│       ├── prime_factors(integer: int, dictionary: bool, pollard_rho: bool) -> typing.Union[list, dict]
│       ├── primes(limit: int) -> list
│       ├── semiprimes(limit: int) -> list
│       ├── split(iterable: typing.Union[list, tuple, str], key: typing.Union[list, tuple], retain: bool) -> list
│       └── twinprimes(limit: int) -> list
├── types
│   ├── CLASS
│   └── FUNCTION
├── ufuncs
│   ├── CLASS
│   └── FUNCTION
│       ├── add(x: Any, y: Any) -> Any
│       ├── apply(a: Any, func: Any, rtype: Any) -> Any
│       ├── base_ufunc(arrays: Any, func: Any, args: Any, rtype: Any) -> Any
│       ├── divide(x: Any, y: Any) -> Any
│       ├── eq(x: Any, y: Any) -> Any
│       ├── floor_divide(x: Any, y: Any) -> Any
│       ├── ge(x: Any, y: Any) -> Any
│       ├── gt(x: Any, y: Any) -> Any
│       ├── le(x: Any, y: Any) -> Any
│       ├── lt(x: Any, y: Any) -> Any
│       ├── modulo(x: Any, y: Any) -> Any
│       ├── multiply(x: Any, y: Any) -> Any
│       ├── ne(x: Any, y: Any) -> Any
│       ├── power(x: Any, y: Any, m: Any) -> Any
│       ├── subtract(x: Any, y: Any) -> Any
│       └── ufunc_helper(x: Any, y: Any, func: Any) -> Any
├── utils
│   ├── CLASS
│   │   ├── InfIterator(object)/__init__(self: Any, start: typing.Union[int, float, complex], mode: str, common: typing.Union[int, float, complex]) -> Any
│   │   ├── IntervalSet(object)/__init__(self: Any, intervals: Any) -> Any
│   │   ├── LinkedList(object)/__init__(self: Any) -> Any
│   │   ├── LinkedListNode(object)/__init__(self: Any, value: Any, next_node: Any) -> Any
│   │   └── OrderedSet(object)/__init__(self: Any, sequence: Any) -> Any
│   └── FUNCTION
└── zh_cn
    ├── CLASS
    └── FUNCTION
        ├── Fraction转为Decimal(分数对象: fractions.Fraction, 有效位数: int) -> decimal.Decimal
        ├── RC4伪随机生成算法(密钥序列: list) -> Any
        ├── RC4初始化密钥调度算法(密钥: bytes) -> list
        ├── RC4密码(文本: bytes, 密钥: bytes) -> bytes
        ├── ROT13密码(文本: str) -> str
        ├── S型函数(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── base64密码(文本: str, 解密: bool) -> str
        ├── x对数y乘积(x: float, y: float) -> float
        ├── y次方根(被开方数: typing.Union[int, float, complex], 开方数: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
        ├── 一维傅里叶变换(数据: Any) -> pypynum.FourierT.FT1D
        ├── 上伽玛(s: typing.Union[int, float, complex], x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
        ├── 上标转整数(上标字符串: str) -> str
        ├── 下伽玛(s: typing.Union[int, float, complex], x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
        ├── 下标转整数(下标字符串: str) -> str
        ├── 中位数(数据: typing.List[float]) -> float
        ├── 中国剩余定理(n: typing.List[int], a: typing.List[int]) -> int
        ├── 中心矩(数据: typing.List[float], 阶数: int) -> float
        ├── 乘积和(数组: typing.List[typing.Any]) -> float
        ├── 代替密码(文本: str, 替换映射: dict, 解密: bool) -> str
        ├── 众数(数据: typing.List[typing.Any]) -> Any
        ├── 伽玛函数(alpha: float) -> float
        ├── 余切(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 余割(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 余弦(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 偏度(数据: typing.List[float]) -> float
        ├── 全一(形状: Any, 返回类型: Any) -> Any
        ├── 全部填充(形状: Any, 填充值: Any, 返回类型: Any) -> Any
        ├── 全零(形状: Any, 返回类型: Any) -> Any
        ├── 写入(文件: str, 对象: object) -> Any
        ├── 几何平均数(数据: typing.List[float]) -> float
        ├── 凯撒密码(文本: str, 移位: int, 解密: bool) -> str
        ├── 分位数(数据: list, 分位值: float, 插值方法: str, 已排序: bool) -> float
        ├── 判定系数(x: typing.List[float], y: typing.List[float]) -> float
        ├── 判断平方数(n: int) -> bool
        ├── 加权平均(数据: typing.List[float], 权重: typing.List[float]) -> float
        ├── 协方差(x: typing.List[float], y: typing.List[float], 自由度: int) -> float
        ├── 原根(a: int, 单个: bool) -> typing.Union[int, typing.List[int]]
        ├── 原点矩(数据: typing.List[float], 阶数: int) -> float
        ├── 双曲余切(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 双曲余割(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 双曲余弦(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 双曲正切(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 双曲正割(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 双曲正弦(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 反余切(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 反余割(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 反余弦(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 反双曲余切(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 反双曲余割(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 反双曲余弦(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 反双曲正切(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 反双曲正割(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 反双曲正弦(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 反正切(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 反正割(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 反正弦(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 可能是平方数(n: int) -> bool
        ├── 填充序列(形状: Any, 序列: Any, 重复: Any, 填充: Any, 返回类型: Any) -> Any
        ├── 多次方根取整(被开方数: int, 开方数: int) -> int
        ├── 多精度余弦(x: typing.Union[int, float], 有效位数: int) -> decimal.Decimal
        ├── 多精度双曲余弦(x: typing.Union[int, float], 有效位数: int) -> decimal.Decimal
        ├── 多精度双曲正弦(x: typing.Union[int, float], 有效位数: int) -> decimal.Decimal
        ├── 多精度反余弦(x: typing.Union[int, float], 有效位数: int) -> decimal.Decimal
        ├── 多精度反正切(x: typing.Union[int, float], 有效位数: int) -> decimal.Decimal
        ├── 多精度反正弦(x: typing.Union[int, float], 有效位数: int) -> decimal.Decimal
        ├── 多精度圆周率(有效位数: int, 方法: str) -> decimal.Decimal
        ├── 多精度复数(实部: typing.Union[int, float, str, decimal.Decimal], 虚部: typing.Union[int, float, str, decimal.Decimal], 有效位数: int) -> pypynum.multiprec.MPComplex
        ├── 多精度对数(真数: typing.Union[int, float], 底数: typing.Union[int, float], 有效位数: int, 使用内置方法: bool) -> decimal.Decimal
        ├── 多精度方位角(y: typing.Union[int, float], x: typing.Union[int, float], 有效位数: int) -> decimal.Decimal
        ├── 多精度欧拉数(有效位数: int, 方法: str) -> decimal.Decimal
        ├── 多精度正弦(x: typing.Union[int, float], 有效位数: int) -> decimal.Decimal
        ├── 多精度自然对数(真数: typing.Union[int, float], 有效位数: int, 使用内置方法: bool) -> decimal.Decimal
        ├── 多精度自然指数(指数: typing.Union[int, float], 有效位数: int, 使用内置方法: bool) -> decimal.Decimal
        ├── 多精度菲涅耳余弦积分(x: typing.Union[int, float], 有效位数: int) -> decimal.Decimal
        ├── 多精度菲涅耳正弦积分(x: typing.Union[int, float], 有效位数: int) -> decimal.Decimal
        ├── 多精度黄金分割率(有效位数: int, 方法: str) -> decimal.Decimal
        ├── 多项式方程(系数: list) -> list
        ├── 字符串转整数(字符串: str) -> int
        ├── 导数(函数: Any, 参数: float, 步长: float, 额外参数: Any, 额外关键字参数: Any) -> float
        ├── 峰度(数据: typing.List[float], 费希尔: bool) -> float
        ├── 希尔256密码(文本: bytes, 密钥: list, 解密: bool) -> bytes
        ├── 平均数(数据: typing.List[float]) -> float
        ├── 平方平均数(数据: typing.List[float]) -> float
        ├── 平方根取整(被开方数: int) -> int
        ├── 序列滚动(序列: typing.Iterator[typing.Any], 偏移: int) -> typing.Iterator[typing.Any]
        ├── 归一化(数据: typing.List[float], 目标: float) -> typing.List[float]
        ├── 扩展欧几里得算法(a: int, b: int) -> typing.Tuple[int, int, int]
        ├── 拆分浮点数字符串(字符串: str) -> tuple
        ├── 排列数(总数: int, 选取数: int) -> int
        ├── 数组(数据: list, 检查: bool) -> pypynum.Array.Array
        ├── 整数转上标(标准字符串: str) -> str
        ├── 整数转下标(标准字符串: str) -> str
        ├── 整数转单词(整数: int) -> str
        ├── 整数转罗马数(整数: int, 上划线: bool) -> str
        ├── 方差(数据: typing.List[float], 自由度: int) -> float
        ├── 普莱费尔密码(文本: str, 密钥: str, 解密: bool) -> str
        ├── 最大公约数(args: int) -> int
        ├── 最小公倍数(args: int) -> int
        ├── 极差(数据: typing.List[float]) -> float
        ├── 标准差(数据: typing.List[float], 自由度: int) -> float
        ├── 模运算阶(a: int, n: int, b: int) -> int
        ├── 欧拉函数(n: int) -> int
        ├── 正切(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 正割(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 正弦(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 浮点数转分数(数值: float, 是否带分数: bool, 误差: float) -> tuple
        ├── 相关系数(x: typing.List[float], y: typing.List[float]) -> float
        ├── 积分(函数: Any, 积分开始: float, 积分结束: float, 积分点数: int, 额外参数: Any, 额外关键字参数: Any) -> float
        ├── 积累乘积(数据: typing.List[float]) -> float
        ├── 符号函数(x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
        ├── 类似形状全一(数组A: Any, 返回类型: Any) -> Any
        ├── 类似形状全零(数组A: Any, 返回类型: Any) -> Any
        ├── 类似形状填充(数组A: Any, 填充值: Any, 返回类型: Any) -> Any
        ├── 累乘积(序列: typing.List[float]) -> typing.List[float]
        ├── 累加和(序列: typing.List[float]) -> typing.List[float]
        ├── 线性方程组(左边: list, 右边: list) -> list
        ├── 组合数(总数: int, 选取数: int) -> int
        ├── 维吉尼亚密码(文本: str, 密钥: str, 解密: bool) -> str
        ├── 罗马数转整数(罗马数: str) -> int
        ├── 自然对数(真数: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 自然指数(指数: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 莫尔斯密码(文本: str, 解密: bool) -> str
        ├── 解析浮点数字符串(字符串: str) -> tuple
        ├── 误差函数(x: typing.Union[int, float]) -> typing.Union[int, float]
        ├── 读取(文件: str) -> list
        ├── 调和平均数(数据: typing.List[float]) -> float
        ├── 贝塔函数(p: float, q: float) -> float
        ├── 贝塞尔函数I0(x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
        ├── 贝塞尔函数I1(x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
        ├── 贝塞尔函数Iv(v: typing.Union[int, float], x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
        ├── 贝塞尔函数J0(x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
        ├── 贝塞尔函数J1(x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
        ├── 贝塞尔函数Jv(v: typing.Union[int, float], x: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
        ├── 负一整数次幂(指数: int) -> int
        ├── 转为多精度复数(实部: typing.Union[int, float, str, decimal.Decimal, complex, pypynum.multiprec.MPComplex], 虚部: typing.Union[int, float, str, decimal.Decimal], 有效位数: int) -> pypynum.multiprec.MPComplex
        ├── 转换为列表(数据: Any) -> list
        ├── 转换为数组(数据: Any) -> pypynum.Array.Array
        ├── 连续乘积(下界: int, 上界: int, 函数: typing.Callable) -> float
        ├── 连续加和(下界: int, 上界: int, 函数: typing.Callable) -> float
        ├── 阶乘函数(n: int) -> int
        ├── 阿特巴什密码(文本: str) -> str
        ├── 频率统计(数据: typing.List[typing.Any]) -> typing.Dict[typing.Any, int]
        └── 黎曼函数(alpha: float) -> float
```

### Code testing

```python
from pypynum import (Array, Geometry, Logic, Matrix, Quaternion, Symbolics, Tensor, Vector,
                     cipher, constants, equations, maths, plotting, random, regression, tools)

...

print(Array.array())
print(Array.array([1, 2, 3, 4, 5, 6, 7, 8]))
print(Array.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
print(Array.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

"""
[]
[1 2 3 4 5 6 7 8]
[[1 2 3 4]
 [5 6 7 8]]
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
"""

triangle = Geometry.Triangle((0, 0), (2, 2), (3, 0))
print(triangle.perimeter())
print(triangle.area())
print(triangle.centroid())

"""
8.06449510224598
3.0
(1.6666666666666667, 0.6666666666666666)
"""

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

"""
sum: [1], carry: [1]
"""

m0 = Matrix.mat([[1, 2], [3, 4]])
m1 = Matrix.mat([[5, 6], [7, 8]])
print(m0)
print(m1)
print(m0 + m1)
print(m0 @ m1)
print(m0.inv())
print(m1.rank())

"""
[[1 2]
 [3 4]]
[[5 6]
 [7 8]]
[[ 6  8]
 [10 12]]
[[19 22]
 [43 50]]
[[ -1.9999999999999996   0.9999999999999998]
 [  1.4999999999999998 -0.49999999999999994]]
2
"""

q0 = Quaternion.quat(1, 2, 3, 4)
q1 = Quaternion.quat(5, 6, 7, 8)
print(q0)
print(q1)
print(q0 + q1)
print(q0 * q1)
print(q0.inverse())
print(q1.conjugate())

"""
(1+2i+3j+4k)
(5+6i+7j+8k)
(6+8i+10j+12k)
(-60+12i+30j+24k)
(0.18257418583505536+-0.3651483716701107i+-0.5477225575051661j+-0.7302967433402214k)
(5+-6i+-7j+-8k)
"""

print(Symbolics.BASIC)
print(Symbolics.ENGLISH)
print(Symbolics.GREEK)
print(Symbolics.parse_expr("-(10+a-(3.14+b0)*(-5))**(-ζn1-2.718/mΣ99)//9"))

"""
%()*+-./0123456789
ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω
[['10', '+', 'a', '-', ['3.14', '+', 'b0'], '*', '-5'], '**', ['-ζn1', '-', '2.718', '/', 'mΣ99'], '//', '9']
"""

t0 = Tensor.ten([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
t1 = Tensor.ten([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
print(t0)
print(t1)
print(t0 + t1)
print(t0 @ t1)

"""
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
[[[ 9 10]
  [11 12]]

 [[13 14]
  [15 16]]]
[[[10 12]
  [14 16]]

 [[18 20]
  [22 24]]]
[[[ 31  34]
  [ 71  78]]

 [[155 166]
  [211 226]]]
"""

string = "PyPyNum"
encrypted = cipher.caesar(string, 10)
print(string)
print(encrypted)
print(cipher.caesar(encrypted, 10, decrypt=True))
encrypted = cipher.vigenere(string, "cipher")
print(string)
print(encrypted)
print(cipher.vigenere(encrypted, "cipher", decrypt=True))
encrypted = cipher.morse(string)
print(string)
print(encrypted)
print(cipher.morse(encrypted, decrypt=True))

"""
PyPyNum
ZiZiXew
PyPyNum
PyPyNum
RgEfRlo
PyPyNum
PyPyNum
.--. -.-- .--. -.-- -. ..- --
PYPYNUM
"""

v0 = Vector.vec([1, 2, 3, 4])
v1 = Vector.vec([5, 6, 7, 8])
print(v0)
print(v1)
print(v0 + v1)
print(v0 @ v1)
print(v0.normalize())
print(v1.angles())

"""
[1 2 3 4]
[5 6 7 8]
[ 5 12 21 32]
70
[0.18257418583505536  0.3651483716701107  0.5477225575051661  0.7302967433402214]
[1.1820279130506308, 1.0985826410133916, 1.0114070854293842, 0.9191723423169716]
"""

print(constants.TB)
print(constants.e)
print(constants.h)
print(constants.phi)
print(constants.pi)
print(constants.tera)

"""
1099511627776
2.718281828459045
6.62607015e-34
1.618033988749895
3.141592653589793
1000000000000
"""

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

"""
[(-1.5615528128088307-6.5209667308287455e-24j), (1.0000000000000007+3.241554513744382e-25j), (2.5615528128088294+4.456233626665941e-24j)]
[1.6666666666666665, -0.6666666666666666, -0.4444444444444444]
"""

print(maths.cot(constants.pi / 3))
print(maths.gamma(1.5))
print(maths.pi(1, 10, lambda x: x ** 2))
print(maths.product([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))
print(maths.sigma(1, 10, lambda x: x ** 2))
print(maths.var([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))

"""
0.577350269189626
0.886226925452758
13168189440000
6469693230
385
73.29
"""

plt = plotting.unary(lambda x: x ** 2, top=10, bottom=0, character="+")
print(plt)
print(plotting.binary(lambda x, y: x ** 2 + y ** 2 - 10, right=10, left=0, compare="<=", basic=plotting.change(plt)))
print(plotting.c_unary(lambda x: x ** x, right=2, left=-2, top=2, bottom=-2, complexity=20, character="-"))

"""
  1.00e+01|         +                               +         
          |                                                   
          |          +                             +          
          |                                                   
          |           +                           +           
          |            +                         +            
          |                                                   
          |             +                       +             
  5.00e+00|_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
          |              +                     +              
          |               +                   +               
          |                +                 +                
          |                 +               +                 
          |                  +             +                  
          |                   +           +                   
          |                    +         +                    
          |                     +++   +++                     
  0.00e+00|________________________+++________________________
           -5.00e+00             0.00e+00             5.00e+00
  1.00e+01|         +                               +         
          |                                                   
          |          +                             +          
          |                                                   
          |.........  +                           +           
          |.............                         +            
          |..............                                     
          |................                     +             
  5.00e+00|................_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
          |................                    +              
          |................                   +               
          |..............  +                 +                
          |.............    +               +                 
          |.........         +             +                  
          |                   +           +                   
          |                    +         +                    
          |                     +++   +++                     
  0.00e+00|________________________+++________________________
           -5.00e+00             0.00e+00             5.00e+00
  2.00e+00|           -                 -           -          -          -            -    
          |               -  -            -          -         -         -           -      
          |                     -           -         -        -        -          -        
          |-                       -          -       -       -        -         -          
          |     -   -                - -       --      -      -       -        -            
          |            -  -              -       -      -     -      -       -             -
          |                  -  - -       - --  - ---  -- -  --     -     - -         - -   
          |                         - -   -  --    --    -   -  - --     -       - -        
          |  -   -  - - -  -          - -- -   ---  ---  -   -   ---   --     - -           
          |             -    -  - - - --    ----- -- -- --- --  --  ---    --           -  -
          |               - -      -     ------------ ----  - --  -- - ---       - - -      
          |    -  -  -  - -  ----- - -- ----------------------- -- ----  - -- --            
          |   -  -   - -         - ---- ---------------------------------      - - - - -  - 
  0.00e+00|_ _ _ _ _ _ _ _-_-_-_-_---- ------------------------------------_-- _ _ _ _ _ _ _
          |            -  -   - - ----------------------------------------- -- - - - -      
          |   -  --  -  -       -- -  -  --------------------------------- -           -  - 
          |    -          - ---- - - -- --------------------- ----- ----    - -- -          
          |               -         - -- --------- -- -- -  -----  ---  -- -       - -  -   
          |             -  - -  - - - -    ---- --- --- --- --  --  ---     - -            -
          |  -   -  - -               - --     --   --   -   -    --   --       --          
          |                       - -     -  --    -    --   -- -  -     --        -  -     
          |                  -  -         - -   - - -  -- -   -     --      -           -   
          |            -  -            - -      --     --     -      -       - -           -
          |     -   -                -         -       -      -       -          -          
          |-                    -  -          -       -        -       -           -        
          |                  -              -         -        -        -            -      
          |               -               -          -         -         -                  
 -2.00e+00|___________-_________________-___________-_____________________-____________-____
           -2.00e+00                            0.00e+00                            2.00e+00
"""

print(random.gauss(0, 1, [2, 3, 4]))
print(random.rand([2, 3, 4]))
print(random.randint(0, 9, [2, 3, 4]))
print(random.uniform(0, 9, [2, 3, 4]))

"""
[[[0.015128082827448793, -0.731558889632968, -0.23379102528494308, 0.5923285646572862], [0.6389462900078073, -1.6347914510943111, 2.3694029836271726, -0.568526047386569], [-1.4229328154353735, 0.45185125607678145, -0.4003256267251042, -1.1425679894907612]], [[1.2876668616276734, 0.934232416262927, -1.4096609242818299, 0.2683613962988281], [0.3503627719719857, 1.9613965063102903, -2.0790609695353077, -0.10339725500993839], [-0.9334087233797456, 1.1394611182611, 1.3341558691128073, -0.3838574172857678]]]
[[[0.8274205130045614, 0.27524584776494854, 0.715710895889572, 0.5807271906102146], [0.21742840470887725, 0.04577819370109826, 0.873689463957162, 0.04119770233167375], [0.554823367037196, 0.5901404246422433, 0.21342393541488192, 0.2979716283166385]], [[0.6045948602408673, 0.265586003384665, 0.9646655285283718, 0.9873208424367568], [0.16916505841642293, 0.15942804932580645, 0.679004396069304, 0.4586819952716237], [0.6058239213086706, 0.37021967026096103, 0.0015603885735545608, 0.8432925281217005]]]
[[[7, 2, 9, 7], [0, 5, 1, 3], [9, 1, 0, 2]], [[1, 2, 7, 5], [5, 7, 4, 1], [2, 5, 7, 9]]]
[[[1.6682230173222767, 0.5174279535822173, 6.202024157209834, 5.097176032335483], [3.44538825088208, 3.7119354081208025, 4.584800897579607, 8.294514147889751], [7.201908571787272, 4.96544760729807, 5.896259095293225, 3.215472062129558]], [[6.352678024277219, 6.894646335413341, 2.0445980257056333, 1.5835361381716893], [6.363077167625872, 8.831103031792672, 6.229821243776864, 0.5639371628314593], [7.639545178199688, 8.079077083978365, 8.063058392021144, 8.673394953496695]]]
"""

print(regression.lin_reg(list(range(5)), [2, 4, 6, 7, 8]))
print(regression.par_reg(list(range(5)), [2, 4, 6, 7, 8]))
print(regression.poly_reg(list(range(5)), [2, 4, 6, 7, 8], 4))

"""
[1.5, 2.4000000000000004]
[-0.21428571428571563, 2.3571428571428625, 1.971428571428569]
[0.08333333333320592, -0.666666666666571, 1.4166666666628345, 1.1666666666688208, 1.9999999999999258]
"""

print(tools.classify([1, 2.3, 4 + 5j, "string", list, True, 3.14, False, tuple, tools]))
print(tools.dedup(["Python", 6, "NumPy", int, "PyPyNum", 9, "pypynum", "NumPy", 6, True]))
print(tools.frange(0, 3, 0.4))
print(tools.linspace(0, 2.8, 8))

"""
{<class 'int'>: [1], <class 'float'>: [2.3, 3.14], <class 'complex'>: [(4+5j)], <class 'str'>: ['string'], <class 'type'>: [<class 'list'>, <class 'tuple'>], <class 'bool'>: [True, False], <class 'module'>: [<module 'pypynum.tools' from 'C:\\Users\\Administrator\\PycharmProjects\\pythonProject\\pypynum\\tools.py'>]}
['Python', 6, 'NumPy', <class 'int'>, 'PyPyNum', 9, 'pypynum', True]
[0.0, 0.4, 0.8, 1.2000000000000002, 1.6, 2.0, 2.4000000000000004, 2.8000000000000003]
[0.0, 0.39999999999999997, 0.7999999999999999, 1.2, 1.5999999999999999, 1.9999999999999998, 2.4, 2.8]
"""

# Tip:
# The test has been successfully passed and ended.
# These tests are only part of the functionality of this package.
# More features need to be explored and tried by yourself!
```
