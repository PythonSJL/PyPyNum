﻿# <font color = blue>PyPyNum</font>

<font color = gree>A multifunctional mathematical calculation package written in pure Python programming language
</font><font color = red>[Python>=3.4]</font>

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

## Version -> 1.9.1 | PyPI -> https://pypi.org/project/PyPyNum/ | Gitee -> https://www.gitee.com/PythonSJL/PyPyNum | GitHub -> https://github.com/PythonSJL/PyPyNum

![LOGO](PyPyNum.png)

PyPI上无法显示logo，可以在Gitee或者GitHub中查看。

The logo cannot be displayed on PyPI, it can be viewed in Gitee or GitHub.

### 介绍

#### Introduction

+ 多功能数学库，类似于numpy、scipy等，专为PyPy解释器制作，亦支持其他类型的Python解释器
+ Multi functional math library, similar to numpy, scipy, etc., designed specifically for PyPy interpreters and also
  supports other types of Python interpreters
+ 不定期更新版本，增加更多实用功能
+ Update versions periodically to add more practical features
+ 如需联系，请添加QQ号2261748025 （Py𝙿𝚢𝚝𝚑𝚘𝚗-水晶兰）
+ If you need to contact, please add QQ number 2261748025 (Py𝙿𝚢𝚝𝚑𝚘𝚗-水晶兰)

### 子模块的名称与功能简介

#### Name and Function Introduction of Submodules

| 子模块名称 Submodule Name  | 功能简介 Function Introduction                 |
|-----------------------|--------------------------------------------|
| `pypynum.Array`       | 多维数组 Multidimensional array                |
| `pypynum.chars`       | 特殊数学符号 Special mathematical symbols        |
| `pypynum.cipher`      | 加密解密算法 Encryption and decryption algorithm |
| `pypynum.constants`   | 数学常数集合 Set of mathematical constants       |
| `pypynum.equations`   | 方程求解 Solving equations                     |
| `pypynum.errors`      | 异常对象 Exception object                      |
| `pypynum.file`        | 文件读写 File read and write                   |
| `pypynum.FourierT`    | 傅里叶变换 Fourier transform                    |
| `pypynum.Geometry`    | 几何形状 Geometric shape                       |
| `pypynum.Graph`       | 图论算法 Graph Theory Algorithm                |
| `pypynum.Group`       | 群论算法 Group Theory Algorithm                |
| `pypynum.Logic`       | 逻辑电路设计 Logic circuit design                |
| `pypynum.maths`       | 通用数学函数 General mathematical functions      |
| `pypynum.Matrix`      | 矩阵运算 Matrix operation                      |
| `pypynum.NeuralN`     | 神经网络训练 Neural network training             |
| `pypynum.numbers`     | 数字处理 Number processing                     |
| `pypynum.plotting`    | 数据可视化 Data visualization                   |
| `pypynum.polynomial`  | 多项式运算 Polynomial operation                 |
| `pypynum.probability` | 概率统计 Probability statistics                |
| `pypynum.Quaternion`  | 四元数运算 Quaternion operation                 |
| `pypynum.random`      | 随机数生成 Random number generation             |
| `pypynum.regression`  | 回归分析 Regression analysis                   |
| `pypynum.sequence`    | 数列计算 Sequence calculation                  |
| `pypynum.Symbolics`   | 符号计算 Symbol calculation                    |
| `pypynum.Tensor`      | 张量运算 Tensor operation                      |
| `pypynum.test`        | 简易测试 Easy test                             |
| `pypynum.this`        | 项目之禅 Zen of Projects                       |
| `pypynum.tools`       | 辅助函数 Auxiliary functions                   |
| `pypynum.Tree`        | 树形数据结构 Tree data structure                 |
| `pypynum.types`       | 特殊类型 Special types                         |
| `pypynum.utils`       | 实用工具 Utility                               |
| `pypynum.Vector`      | 向量运算 Vector operation                      |

### PyPyNum的Zen（预览）

#### The Zen of PyPyNum (Preview)

```
    The Zen of PyPyNum, by Shen Jiayi

This is a math package written purely in Python.

Elegant is superior to clunky.
Clarity trumps obscurity.
Straightforwardness is preferred over convolution.
Sophisticated is better than overcomplicated.
Flat structure beats nested hierarchies.
Sparse code wins over bloated ones.
```

```
...

Do you want to view all the content?

Enter "from pypynum import this" in your

Python interpreter and run it!
```

```
                                        February 27, 2024
```

### 与上一个版本相比新增功能

#### New features compared to the previous version

```
!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=

修复了一些功能问题

Fixed some functional issues

!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=
```

### 运行用时测试

#### Run Time Test

|                     矩阵用时测试<br>Matrix Time Test                     |                                                                            NumPy﻿+﻿CPython﻿（﻿seconds﻿）                                                                             | 排名<br>Ranking |                                                                             PyPyNum﻿+﻿PyPy﻿（﻿seconds﻿）                                                                             | 排名<br>Ranking |                                                                           Mpmath﻿_﻿+﻿_﻿PyPy﻿_﻿（﻿_﻿seconds﻿_﻿）                                                                           | 排名<br>Ranking |                                                                                                     SymPy﻿_﻿+﻿_﻿PyPy﻿_﻿（﻿_﻿seconds﻿_﻿）                                                                                                     | 排名<br>Ranking |
|:------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------:|
| 创﻿建﻿一﻿百﻿阶﻿随﻿机﻿数﻿矩﻿阵<br>Create a hundred order random number matrix |                                                                                      0.000083                                                                                      |       1       |                                                                                      0.005374                                                                                      |       2       |                                                                                        0.075253                                                                                         |       3       |                                                                                                                  0.230530                                                                                                                  |       4       |
|     创建一千阶随机数矩阵<br>Create a thousand order random number matrix     |                                                                                      0.006740                                                                                      |       1       |                                                                                      0.035666                                                                                      |       2       |                                                                                        1.200950                                                                                         |       3       |                                                                                                                  4.370265                                                                                                                  |       4       |
|        一百阶矩阵相加<br>Addition of matrices of order one hundred        |                                                                                      0.000029                                                                                      |       1       |                                                                                      0.002163                                                                                      |       2       |                                                                                        0.045641                                                                                         |       4       |                                                                                                                  0.035700                                                                                                                  |       3       |
|          一千阶矩阵相加<br>Adding matrices of order one thousand          |                                                                                      0.002647                                                                                      |       1       |                                                                                      0.019111                                                                                      |       2       |                                                                                        1.746957                                                                                         |       4       |                                                                                                                  0.771542                                                                                                                  |       3       |
|         一百阶矩阵行列式<br>Determinant of a hundred order matrix          |                                                                                      0.087209                                                                                      |       2       |                                                                                      0.016331                                                                                      |       1       |                                                                                        4.354507                                                                                         |       3       |                                                                                                                  5.157206                                                                                                                  |       4       |
|         一千阶矩阵行列式<br>Determinant of a thousand order matrix         |                                                                                      0.616113                                                                                      |       1       |                                                                                      3.509747                                                                                      |       2       |                                                                                          速度极慢                                                                                           |       3       |                                                                                                                    无法计算                                                                                                                    |       4       |
|      一百阶矩阵求逆<br>Finding the inverse of a hundred order matrix      |                                                                                      0.162770                                                                                      |       1       |                                                                                     31.088849                                                                                      |       4       |                                                                                        8.162948                                                                                         |       2       |                                                                                                                 21.437424                                                                                                                  |       3       |
|     一千阶矩阵求逆<br>Finding the inverse of a thousand order matrix      |                                                                                      0.598905                                                                                      |       1       |                                                                                        速度较慢                                                                                        |       4       |                                                                                          速度较慢                                                                                           |       2       |                                                                                                                    速度较慢                                                                                                                    |       3       |
|                   数组输出效果<br>Array output effect                    | ```[[[[ -7 -67]```<br>```[-78  29]]```<br><br>```[[-86 -97]```<br>```[ 68  -3]]]```<br><br><br>```[[[ 11  42]```<br>```[ 24 -65]]```<br><br>```[[-60  72]```<br>```[ 73   2]]]]``` |       /       | ```[[[[ 37  83]```<br>```[ 40   2]]```<br><br>```[[ -5 -34]```<br>```[ -7  72]]]```<br><br><br>```[[[ 13 -64]```<br>```[  6  90]]```<br><br>```[[ 68  57]```<br>```[ 78  11]]]]``` |       /       | ```[-80.0   -8.0  80.0  -88.0]```<br>```[-99.0  -43.0  87.0   81.0]```<br>```[ 20.0  -55.0  98.0    8.0]```<br>```[  8.0   44.0  64.0  -35.0]```<br>(只支持矩阵)<br>(Only supports matrices) |       /       | ```⎡⎡16   -56⎤  ⎡ 8   -28⎤⎤```<br>```⎢⎢        ⎥  ⎢        ⎥⎥```<br>```⎢⎣-56  56 ⎦  ⎣-28  28 ⎦⎥```<br>```⎢                      ⎥```<br>```⎢ ⎡-2  7 ⎤   ⎡-18  63 ⎤⎥```<br>```⎢ ⎢      ⎥   ⎢        ⎥⎥```<br>```⎣ ⎣7   -7⎦   ⎣63   -63⎦⎦``` |       /       |

### 基本结构

#### Basic structure

```
PyPyNum
├── Array
│   ├── CLASS
│   │   └── Array(object)/__init__(self: Any, data: Any, check: Any) -> Any
│   └── FUNCTION
│       ├── array(data: Any) -> Any
│       ├── fill(shape: Any, sequence: Any, repeat: Any, pad: Any) -> Any
│       ├── function(_array: Any, _function: Any, args: Any) -> Any
│       ├── get_shape(data: Any) -> Any
│       ├── is_valid_array(_array: Any, _shape: Any) -> Any
│       ├── zeros(shape: Any) -> Any
│       └── zeros_like(_nested_list: Any) -> Any
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
│   │   └── Group(object)/__init__(self: Any, data: Any) -> Any
│   └── FUNCTION
│       ├── add(x: Any, y: Any) -> Any
│       ├── divide(x: Any, y: Any) -> Any
│       ├── group(data: Any) -> Any
│       ├── multiply(x: Any, y: Any) -> Any
│       └── subtract(x: Any, y: Any) -> Any
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
│       ├── eigen(matrix: pypynum.Matrix.Matrix) -> tuple
│       ├── hessenberg(matrix: pypynum.Matrix.Matrix) -> tuple
│       ├── identity(n: int) -> pypynum.Matrix.Matrix
│       ├── lu(matrix: pypynum.Matrix.Matrix) -> tuple
│       ├── mat(data: Any) -> Any
│       ├── qr(matrix: pypynum.Matrix.Matrix) -> tuple
│       ├── rotate90(matrix: pypynum.Matrix.Matrix, times: int) -> pypynum.Matrix.Matrix
│       ├── same(rows: Any, cols: Any, value: Any) -> Any
│       ├── svd(matrix: pypynum.Matrix.Matrix) -> tuple
│       ├── tril_indices(n: int, k: int, m: int) -> tuple
│       ├── zeros(_dimensions: Any) -> Any
│       └── zeros_like(_nested_list: Any) -> Any
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
│       ├── same(length: Any, value: Any) -> Any
│       ├── vec(data: Any) -> Any
│       ├── zeros(_dimensions: Any) -> Any
│       └── zeros_like(_nested_list: Any) -> Any
├── chars
│   ├── CLASS
│   └── FUNCTION
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
├── constants
│   ├── CLASS
│   └── FUNCTION
├── equations
│   ├── CLASS
│   └── FUNCTION
│       ├── linear_equation(left: list, right: list) -> list
│       └── polynomial_equation(coefficients: list) -> list
├── errors
│   ├── CLASS
│   └── FUNCTION
├── file
│   ├── CLASS
│   └── FUNCTION
│       ├── read(file: str) -> list
│       └── write(file: str, cls: object) -> Any
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
│       ├── average(data: Any, weights: Any, expected: Any) -> Any
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
│       ├── definite_integral(f: Any, x_start: typing.Union[int, float], x_end: typing.Union[int, float], n: int, args: Any, kwargs: Any) -> float
│       ├── derivative(f: Any, x: typing.Union[int, float], h: typing.Union[int, float], args: Any, kwargs: Any) -> float
│       ├── erf(x: typing.Union[int, float]) -> float
│       ├── exgcd(a: int, b: int) -> tuple
│       ├── exp(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── factorial(n: int) -> int
│       ├── freq(data: typing.Union[list, tuple]) -> dict
│       ├── gamma(alpha: typing.Union[int, float]) -> float
│       ├── gaussian(x: typing.Union[int, float], _mu: typing.Union[int, float], _sigma: typing.Union[int, float]) -> float
│       ├── gcd(args: int) -> int
│       ├── geom_mean(numbers: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── harm_mean(numbers: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── iroot(y: int, n: int) -> int
│       ├── is_possibly_square(n: int) -> bool
│       ├── is_square(n: int) -> bool
│       ├── isqrt(x: int) -> int
│       ├── kurt(data: typing.Union[list, tuple]) -> float
│       ├── lcm(args: int) -> int
│       ├── ln(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── mean(numbers: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── median(numbers: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── mod_order(a: int, n: int, b: int) -> int
│       ├── mode(data: typing.Union[list, tuple]) -> Any
│       ├── normalize(data: typing.Union[list, tuple], target: typing.Union[int, float, complex]) -> typing.Union[list, tuple]
│       ├── parity(x: int) -> int
│       ├── pi(i: int, n: int, f: Any) -> typing.Union[int, float, complex]
│       ├── poisson(x: int, _lambda: typing.Union[int, float]) -> float
│       ├── primitive_root(a: int, single: bool) -> typing.Union[int, list]
│       ├── product(numbers: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── ptp(numbers: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── raw_moment(data: typing.Union[list, tuple], order: int) -> float
│       ├── root(x: typing.Union[int, float, complex], y: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
│       ├── sec(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── sech(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── sigma(i: int, n: int, f: Any) -> typing.Union[int, float, complex]
│       ├── sigmoid(x: typing.Union[int, float]) -> float
│       ├── sign(x: typing.Union[int, float]) -> int
│       ├── sin(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── sinh(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── skew(data: typing.Union[list, tuple]) -> float
│       ├── square_mean(numbers: typing.Union[list, tuple]) -> typing.Union[int, float, complex]
│       ├── std(numbers: typing.Union[list, tuple], dof: int) -> typing.Union[int, float, complex]
│       ├── tan(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── tanh(x: typing.Union[int, float]) -> typing.Union[int, float]
│       ├── totient(n: int) -> int
│       ├── var(numbers: typing.Union[list, tuple], dof: int) -> typing.Union[int, float, complex]
│       └── zeta(alpha: typing.Union[int, float, complex]) -> typing.Union[int, float, complex]
├── numbers
│   ├── CLASS
│   └── FUNCTION
│       ├── float2fraction(number: float, mixed: bool, error: float) -> tuple
│       ├── int2roman(integer: int, overline: bool) -> str
│       ├── roman2int(roman_num: str) -> int
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
│       └── poly(terms: Any) -> Any
├── probability
│   ├── CLASS
│   └── FUNCTION
│       ├── binomial(sample_size: int, successes: int, success_probability: typing.Union[int, float]) -> float
│       ├── chi2_cont(contingency: list, calc_p: bool, corr: bool) -> tuple
│       ├── chi2_pdf(x: typing.Union[int, float], k: typing.Union[int, float]) -> float
│       └── hypergeometric(total_items: int, success_items: int, sample_size: int, successes_in_sample: int) -> float
├── random
│   ├── CLASS
│   └── FUNCTION
│       ├── choice(seq: typing.Union[list, tuple, str], shape: typing.Union[list, tuple]) -> Any
│       ├── gauss(mu: typing.Union[int, float], sigma: typing.Union[int, float], shape: typing.Union[list, tuple]) -> typing.Union[float, list]
│       ├── gauss_error(original: typing.Union[list, tuple], mu: typing.Union[int, float], sigma: typing.Union[int, float]) -> list
│       ├── rand(shape: typing.Union[list, tuple]) -> typing.Union[float, list]
│       ├── randint(a: int, b: int, shape: typing.Union[list, tuple]) -> typing.Union[int, list]
│       └── uniform(a: typing.Union[int, float], b: typing.Union[int, float], shape: typing.Union[list, tuple]) -> typing.Union[float, list]
├── regression
│   ├── CLASS
│   └── FUNCTION
│       ├── linear_regression(x: typing.Union[list, tuple], y: typing.Union[list, tuple]) -> list
│       ├── parabolic_regression(x: typing.Union[list, tuple], y: typing.Union[list, tuple]) -> list
│       └── polynomial_regression(x: typing.Union[list, tuple], y: typing.Union[list, tuple], n: int) -> list
├── sequence
│   ├── CLASS
│   └── FUNCTION
│       ├── arithmetic_sequence(a1: typing.Union[int, float], an: typing.Union[int, float], d: typing.Union[int, float], n: typing.Union[int, float], s: typing.Union[int, float]) -> dict
│       ├── bernoulli(n: int, single: bool) -> list
│       ├── catalan(n: int, single: bool) -> typing.Union[int, list]
│       ├── farey(n: int) -> list
│       ├── fibonacci(n: int, single: bool) -> typing.Union[int, list]
│       ├── geometric_sequence(a1: typing.Union[int, float], an: typing.Union[int, float], r: typing.Union[int, float], n: typing.Union[int, float], s: typing.Union[int, float]) -> dict
│       └── recaman(n: int, single: bool) -> typing.Union[int, list]
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
│       ├── generate_primes(limit: int) -> list
│       ├── generate_semiprimes(limit: int) -> list
│       ├── geomspace(start: typing.Union[int, float], stop: typing.Union[int, float], number: int) -> list
│       ├── interp(data: typing.Union[list, tuple], length: int) -> list
│       ├── linspace(start: typing.Union[int, float], stop: typing.Union[int, float], number: int) -> list
│       ├── magic_square(n: Any) -> Any
│       ├── primality(n: int, iter_num: int) -> bool
│       ├── prime_factors(integer: int, dictionary: bool, pollard_rho: bool) -> typing.Union[list, dict]
│       └── split(iterable: typing.Union[list, tuple, str], key: typing.Union[list, tuple], retain: bool) -> list
├── types
│   ├── CLASS
│   └── FUNCTION
├── ufuncs
│   ├── CLASS
│   └── FUNCTION
│       ├── add(x: Any, y: Any) -> Any
│       ├── divide(x: Any, y: Any) -> Any
│       ├── floor_divide(x: Any, y: Any) -> Any
│       ├── modulo(x: Any, y: Any) -> Any
│       ├── multiply(x: Any, y: Any) -> Any
│       ├── power(x: Any, y: Any, m: Any) -> Any
│       └── subtract(x: Any, y: Any) -> Any
└── utils
    ├── CLASS
    │   ├── InfIterator(object)/__init__(self: Any, start: typing.Union[int, float, complex], mode: str, common: typing.Union[int, float, complex]) -> Any
    │   ├── LinkedList(object)/__init__(self: Any) -> Any
    │   ├── LinkedListNode(object)/__init__(self: Any, value: Any, next_node: Any) -> Any
    │   └── OrderedSet(object)/__init__(self: Any, sequence: Any) -> Any
    └── FUNCTION
```

### 代码测试

#### Code testing

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
[[-2.0  1.0]
 [ 1.5 -0.5]]
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
print(equations.polynomial_equation(p))
print(equations.linear_equation(*m))

"""
[[(-1.5615528128088307-6.5209667308287455e-24j)                                             0                                             0]
 [                                            0   (2.5615528128088294+4.456233626665941e-24j)                                             0]
 [                                            0                                             0   (1.0000000000000007+3.241554513744382e-25j)]]
[ 1.6666666666666667 -0.6666666666666666 -0.4444444444444444]
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
[[[0.4604088962341266, -1.7729143777833498, -1.027113249139529, -1.367335972424437], [1.5152952392963814, 0.1143532276512219, -2.2478367353916626, 0.770010737736378], [-0.4126751204277065, -2.5483288418244814, -0.8493985529797649, -0.5563319772201964]], [[1.9010219123281087, 0.44289739103266357, -0.7148439667426828, -0.742966218100922], [0.21210379525283574, 1.5466635593995341, -0.4536304781981763, 0.5978046752778463], [-0.802299453332161, -1.0295309618075863, 0.5960990076298143, -0.7956578324150254]]]
[[[0.6378288221898551, 0.5667742466043126, 0.4656215108828976, 0.15042085420645113], [0.17290475960349394, 0.7286971110875379, 0.645886619551428, 0.6328537921605502], [0.5626377160935252, 0.41015810474249603, 0.9951566294215863, 0.21679108443018347]], [[0.7747392018966252, 0.5885873225687281, 0.4305122635955937, 0.1102178686782671], [0.6823600514076231, 0.919946966200235, 0.9010988477920265, 0.9324975541841414], [0.026520946245817467, 0.7483826867189314, 0.25799134110551736, 0.7231613737350734]]]
[[[5, 5, 4, 2], [2, 0, 9, 6], [9, 2, 0, 0]], [[0, 3, 9, 2], [2, 5, 5, 3], [9, 2, 5, 1]]]
[[[4.661185893580614, 6.263504213542613, 1.5982048879385133, 4.592524740685044], [5.915777961319595, 0.7073727896327806, 2.6233256174392263, 4.980453415047565], [2.257336580759053, 8.299789041235444, 5.839322602213961, 3.1890466264933925]], [[8.498155134210208, 4.722625882838601, 4.4036516933236705, 4.313235000904077], [3.7418534910533126, 2.7783464437089305, 8.884554145257127, 3.860087039759355], [4.430653527343895, 0.9625175276439548, 6.813933725727357, 7.910773669181097]]]
"""

print(regression.linear_regression(list(range(5)), [2, 4, 6, 7, 8]))
print(regression.parabolic_regression(list(range(5)), [2, 4, 6, 7, 8]))
print(regression.polynomial_regression(list(range(5)), [2, 4, 6, 7, 8], 4))

"""
[1.5, 2.4000000000000004]
[-0.21428571428571563, 2.3571428571428625, 1.971428571428569]
[0.08333333334800574, -0.6666666668092494, 1.4166666678382942, 1.1666666648311956, 2.0000000002900613]
"""

print(tools.classify([1, 2.3, 4 + 5j, "string", list, True, 3.14, False, tuple, tools]))
print(tools.dedup(["Python", 6, "NumPy", int, "PyPyNum", 9, "pypynum", "NumPy", 6, True]))
print(tools.frange(0, 3, 0.4))
print(tools.linspace(0, 2.8, 8))

"""
{<class 'int'>: [1], <class 'float'>: [2.3, 3.14], <class 'complex'>: [(4+5j)], <class 'str'>: ['string'], <class 'type'>: [<class 'list'>, <class 'tuple'>], <class 'bool'>: [True, False], <class 'module'>: [<module 'pypynum.tools' from 'F:\\PyPyproject\\PyPyproject1\\pypynum\\tools.py'>]}
['Python', 6, 'NumPy', <class 'int'>, 'PyPyNum', 9, 'pypynum', True]
[0.0, 0.4, 0.8, 1.2000000000000002, 1.6, 2.0, 2.4000000000000004, 2.8000000000000003]
[0.0, 0.39999999999999997, 0.7999999999999999, 1.2, 1.5999999999999999, 1.9999999999999998, 2.4, 2.8]
"""

# 提示：
# 
# 测试已成功通过并结束。
# 
# 这些测试只是这个包功能的一部分。
# 
# 更多的功能需要自己探索和尝试！
# 
# Tip:
# 
# The test has been successfully passed and ended.
# 
# These tests are only part of the functionality of this package.
# 
# More features need to be explored and tried by yourself!
```
