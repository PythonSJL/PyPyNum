# PyPyNum

## Version -> 1.2.0 | PyPI -> https://pypi.org/project/PyPyNum/ | Gitee -> https://www.gitee.com/PythonSJL/PyPyNum

### 介绍

#### Introduction

+ DIY数学库，类似于numpy、scipy等，专为PyPy解释器制作
+ DIY math library, similar to numpy, scipy, etc., specifically designed for PyPy interpreters
+ 不定期更新版本，增加更多实用功能
+ Update versions periodically to add more practical features
+ 如需联系，QQ 2261748025 （Py𝙿𝚢𝚝𝚑𝚘𝚗-水晶兰）
+ If you need to contact, QQ 2261748025 (Py𝙿𝚢𝚝𝚑𝚘𝚗-水晶兰)

### PyPyNum的Zen

#### The Zen of PyPyNum

```
    The Zen of PyPyNum, by Shen Jiayi

This is a math package written purely in Python.

... (Do you want to see the entire content?
Then enter "from pypynum import this" on your
Python interpreter and run it!)

                                December 27, 2023
```

### 与上一个版本相比新增功能

#### New features compared to the previous version

```
PyPyNum
    Logic
        CLASSES
            Basic
                Binary
                    AND
                    COMP +
                    HalfAdder
                    HalfSuber +
                    JKFF +
                    NAND
                    NOR
                    OR
                    XNOR
                    XOR
                Quaternary +
                    TwoBDiver +
                    TwoBMuler +
                Ternary
                    FullAdder
                    FullSuber +
                Unary
                    DFF +
                    NOT
                    TFF +

("+" indicates newly added)

PyPyNum
    cipher
        FUNCTIONS
            dna(string: str, decrypt: bool = False) -> str

[Code modification and repair]
```

### 基本结构

#### Basic structure

```
PyPyNum
    errors
        CLASSES
            LogicError
            RandomError
            ShapeError
    test
        [A Code Test File]
    this
        [The Zen of PyPyNum]
    types
        DATA
            arr = list | tuple
            ite = list | tuple | str
            num = int | float | complex
            real = int | float
    Array
        CLASSES
            Array
        FUNCTIONS
            array(data=None)
            is_valid_array(_array, _shape)
            zeros(_dimensions)
            zeros_like(_nested_list)
    Geometry
        CLASSES
            Circle
            Line
            Point
            Polygon
            Quadrilateral
            Triangle
        FUNCTIONS
            distance(g1, g2, error: int | float = 0) -> float
    Logic
        CLASSES
            Basic
                Binary
                    AND
                    COMP
                    HalfAdder
                    HalfSuber
                    JKFF
                    NAND
                    NOR
                    OR
                    XNOR
                    XOR
                Quaternary
                    TwoBDiver
                    TwoBMuler
                Ternary
                    FullAdder
                    FullSuber
                Unary
                    DFF
                    NOT
                    TFF
    Matrix
        CLASSES
            Matrix
        FUNCTIONS
            eig(matrix)
            identity(n)
            lu(matrix)
            mat(data)
            qr(matrix)
            same(rows, cols, value=0)
            svd(matrix)
            tril_indices(n, k=0, m=None)
            zeros(_dimensions)
            zeros_like(_nested_list)
    Quaternion
        CLASSES
            Euler
            Quaternion
        FUNCTIONS
            change(data: Euler | Quaternion) -> Quaternion | Euler
            euler(yaw: int | float = 0, pitch: int | float = 0, roll: int | float = 0) -> Euler
            quat(w: int | float = 0, x: int | float = 0, y: int | float = 0, z: int | float = 0) -> Quaternion
    Symbolics
        FUNCTIONS
            interpreter(expr: str) -> list
        DATA
            basic = '%()*+-./0123456789'
            english = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            greek = 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟ∏ΡΣΤΥΦΧΨΩβγδεζηθικλμνξοπρστυφχψω'
            operators = ['**', '*', '//', '/', '%', '+', '-']
            valid = '%()*+-./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcd...yzΑΒΓΔΕΖΗΘ...'
    Tensor
        CLASSES
            Tensor
        FUNCTIONS
            ten(data)
            tensor_and_number(tensor, operator, number)
            tolist(_nested_list)
            zeros(_dimensions)
            zeros_like(_nested_list)
    Vector
        CLASSES
            Vector
        FUNCTIONS
            same(length, value=0)
            vec(data)
            zeros(_dimensions)
            zeros_like(_nested_list)
    cipher
        FUNCTIONS
            dna(string: str, decrypt: bool = False) -> str
    constants
        DATA
            AMU = 1.6605402e-27
            EB = 1152921504606846976
            G = 6.6743e-11
            GB = 1073741824
            KB = 1024
            MB = 1048576
            NA = 6.02214076e+23
            PB = 1125899906842624
            TB = 1099511627776
            YB = 1208925819614629174706176
            ZB = 1180591620717411303424
            atto = 1e-18
            c = 299792458
            centi = 0.01
            deci = 0.1
            deka = 10.0
            e = 2.718281828459045
            exa = 1e+18
            femto = 1e-15
            gamma = 0.5772156649015329
            giga = 1000000000.0
            h = 6.62607015e-34
            hecto = 100.0
            inf = inf
            kilo = 1000.0
            mega = 1000000.0
            micro = 1e-06
            milli = 0.001
            nan = nan
            nano = 1e-09
            peta = 1000000000000000.0
            phi = 1.618033988749895
            pi = 3.141592653589793
            pico = 1e-12
            qe = 1.60217733e-19
            tera = 1000000000000.0
            yocto = 1e-24
            yotta = 1e+24
            zepto = 1e-21
            zetta = 1e+21
    equations
        FUNCTIONS
            mles = multivariate_linear_equation_system(left: list, right: list) -> None | list
            multivariate_linear_equation_system(left: list, right: list) -> None | list
            pe = polynomial_equation(coefficients: list) -> list
            polynomial_equation(coefficients: list) -> list
    mathematics
        FUNCTIONS
            A = arrangement(n: int, r: int) -> int
            C = combination(n: int, r: int) -> int
            acos(x: int | float) -> int | float
            acosh(x: int | float) -> int | float
            acot(x: int | float) -> int | float
            acoth(x: int | float) -> int | float
            acsc(x: int | float) -> int | float
            acsch(x: int | float) -> int | float
            arrangement(n: int, r: int) -> int
            asec(x: int | float) -> int | float
            asech(x: int | float) -> int | float
            asin(x: int | float) -> int | float
            asinh(x: int | float) -> int | float
            atan(x: int | float) -> int | float
            atanh(x: int | float) -> int | float
            beta(p: int | float, q: int | float) -> int | float
            combination(n: int, r: int) -> int
            cos(x: int | float) -> int | float
            cosh(x: int | float) -> int | float
            cot(x: int | float) -> int | float
            coth(x: int | float) -> int | float
            csc(x: int | float) -> int | float
            csch(x: int | float) -> int | float
            definite_integral(f, x_start: int | float, x_end: int | float, n: int = 10000000) -> float
            derivative(f, x: int | float, h: int | float = 1e-07) -> float
            erf(x: int | float) -> float
            exp(x: int | float) -> int | float
            factorial(n: int) -> int
            freq(data: list | tuple) -> dict
            gamma(alpha: int | float) -> float
            gaussian(x: int | float, _mu: int | float = 0, _sigma: int | float = 1) -> float
            ln(x: int | float) -> int | float
            mean(numbers: list | tuple) -> int | float | complex
            median(numbers: list | tuple) -> int | float | complex
            mode(data: list | tuple) -> <built-in function any>
            pi(i: int, n: int, f) -> int | float | complex
            product(numbers: list | tuple) -> int | float | complex
            ptp(numbers: list | tuple) -> int | float | complex
            root(x: int | float | complex, y: int | float | complex) -> int | float | complex
            sec(x: int | float) -> int | float
            sech(x: int | float) -> int | float
            sigma(i: int, n: int, f) -> int | float | complex
            sigmoid(x: int | float) -> float
            sign(x: int | float) -> int
            sin(x: int | float) -> int | float
            sinh(x: int | float) -> int | float
            std(numbers: list | tuple) -> int | float | complex
            tan(x: int | float) -> int | float
            tanh(x: int | float) -> int | float
            var(numbers: list | tuple) -> int | float | complex
            zeta(alpha: int | float) -> float
    neuralnetwork
        CLASSES
            NeuralNetwork
    plotting
        FUNCTIONS
            background(right: int | float = 5, left: int | float = -5, top: int | float = 5, bottom: int | float = -5, complexity: int | float = 5, ratio: int | float = 3, merge: bool = False) -> list | str
            binary(function, right: int | float = 5, left: int | float = -5, top: int | float = 5, bottom: int | float = -5, complexity: int | float = 5, ratio: int | float = 3, error=0, compare='==', merge: bool = True, basic: list = None, character: str = '.', data: bool = False) -> list | str
            c_unary(function, start: int | float, end: int | float, interval: int | float = 5, projection: str = 'ri', right: int | float = 5, left: int | float = -5, top: int | float = 5, bottom: int | float = -5, complexity: int | float = 5, ratio: int | float = 3, merge: bool = True, basic: list = None, character: str = '.', data: bool = False) -> list | str
            change(data: list | str) -> list | str
            unary(function, right: int | float = 5, left: int | float = -5, top: int | float = 5, bottom: int | float = -5, complexity: int | float = 5, ratio: int | float = 3, merge: bool = True, basic: list = None, character: str = '.', data: bool = False) -> list | str
    random
        FUNCTIONS
            choice(seq: list | tuple | str, shape: list | tuple = None)
            gauss(mu: int | float = 0, sigma: int | float = 1, shape: list | tuple = None) -> float | list
            gauss_error(original: list | tuple, mu: int | float = 0, sigma: int | float = 1) -> list
            rand(shape: list | tuple = None) -> float | list
            randint(a: int, b: int, shape: list | tuple = None) -> int | list
            uniform(a: int | float, b: int | float, shape: list | tuple = None) -> float | list
    regression
        FUNCTIONS
            linear_regression(x, y)
            parabolic_regression(x, y)
    tools
        FUNCTIONS
            classify(array: list | tuple) -> dict
            deduplicate(iterable: list | tuple | str) -> list | tuple | str
            frange(start: int | float, stop: int | float, step: float = 1.0) -> list
            linspace(start: int | float, stop: int | float, number: int) -> list
```

### 代码测试

#### Code testing

```
>>> from pypynum import (Array, Geometry, Matrix, Quaternion, Symbolics, Tensor, Vector,
                         cipher, constants, equations, mathematics, plotting, random, regression, tools)

...

>>> print(Array.array())
>>> print(Array.array([1, 2, 3, 4, 5, 6, 7, 8]))
>>> print(Array.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
>>> print(Array.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

[]
[1 2 3 4 5 6 7 8]
[[1 2 3 4]
 [5 6 7 8]]
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]

>>> triangle = Geometry.Triangle((0, 0), (2, 2), (3, 0))
>>> print(triangle.perimeter())
>>> print(triangle.area())
>>> print(triangle.centroid())

8.06449510224598
3.0
(1.6666666666666667, 0.6666666666666666)

>>> m0 = Matrix.mat([[1, 2], [3, 4]])
>>> m1 = Matrix.mat([[5, 6], [7, 8]])
>>> print(m0)
>>> print(m1)
>>> print(m0 + m1)
>>> print(m0 @ m1)
>>> print(m0.inv())
>>> print(m1.rank())

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

>>> q0 = Quaternion.quat(1, 2, 3, 4)
>>> q1 = Quaternion.quat(5, 6, 7, 8)
>>> print(q0)
>>> print(q1)
>>> print(q0 + q1)
>>> print(q0 * q1)
>>> print(q0.inverse())
>>> print(q1.conjugate())

(1+2i+3j+4k)
(5+6i+7j+8k)
(6+8i+10j+12k)
(-60+12i+30j+24k)
(0.18257418583505536+-0.3651483716701107i+-0.5477225575051661j+-0.7302967433402214k)
(5+-6i+-7j+-8k)

>>> print(Symbolics.basic)
>>> print(Symbolics.english)
>>> print(Symbolics.greek)
>>> print(Symbolics.interpreter("-(10+a-(3.14+b0)*(-5))**(-ζn1-2.718/mΣ99)//9"))

%()*+-./0123456789
ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟ∏ΡΣΤΥΦΧΨΩβγδεζηθικλμνξοπρστυφχψω
[['10', '+', 'a', '-', ['3.14', '+', 'b0'], '*', '-5'], '**', ['-ζn1', '-', '2.718', '/', 'mΣ99'], '//', '9']

>>> t0 = Tensor.ten([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
>>> t1 = Tensor.ten([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
>>> print(t0)
>>> print(t1)
>>> print(t0 + t1)
>>> print(t0 @ t1)

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

>>> string = "PyPyNum"
>>> encrypted = cipher.dna(string)
>>> print(string)
>>> print(encrypted)
>>> print(cipher.dna(encrypted, decrypt=True))

PyPyNum
CCCTAGACCCTCGTCCCGCTAAACCCTG
PyPyNum

v0 = Vector.vec([1, 2, 3, 4])
v1 = Vector.vec([5, 6, 7, 8])
print(v0)
print(v1)
print(v0 + v1)
print(v0 @ v1)
print(v0.normalize())
print(v1.angles())

[1 2 3 4]
[5 6 7 8]
[ 5 12 21 32]
70
[0.18257418583505536  0.3651483716701107  0.5477225575051661  0.7302967433402214]
[1.1820279130506308, 1.0985826410133916, 1.0114070854293842, 0.9191723423169716]

>>> print(constants.TB)
>>> print(constants.e)
>>> print(constants.h)
>>> print(constants.phi)
>>> print(constants.pi)
>>> print(constants.tera)

1099511627776
2.718281828459045
6.62607015e-34
1.618033988749895
3.141592653589793
1000000000000.0

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


    提示：Matrix模块的eig函数可能存在计算错误

    Tip: The eig function of the Matrix module may have calculation errors
    
[2.561552812809, -1.561552812809, 1.0]
[1.666666666667, -0.666666666667, -0.444444444444]

>>> print(mathematics.cot(constants.pi / 3))
>>> print(mathematics.gamma(1.5))
>>> print(mathematics.pi(1, 10, lambda x: x ** 2))
>>> print(mathematics.product([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))
>>> print(mathematics.sigma(1, 10, lambda x: x ** 2))
>>> print(mathematics.var([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))

0.577350269189626
0.886226925452758
13168189440000
6469693230
385
73.29

>>> plt = plotting.unary(lambda x: x ** 2, top=10, bottom=0, character="+")
>>> print(plt)
>>> print(plotting.binary(lambda x, y: x ** 2 + y ** 2 - 10, right=10, left=0, compare="<=", basic=plotting.change(plt)))
>>> print(plotting.c_unary(lambda x: x ** x, start=-10, end=10, interval=100, right=2, left=-2, top=2, bottom=-2, complexity=20, character="-"))

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
  2.00e+00|                                                                                 
          |                                                                                 
          |                                                                                 
          |                                                                                 
          |                                                                                 
          |                                                                                 
          |                                                                                 
          |                                                                                 
          |                                                                                 
          |                                -------                                          
          |                          ------       -----                                     
          |                       ----                 --                                   
          |                     ---                     --                                  
  0.00e+00|_ _ _ _ _ _ _ _ _ _ --_ _ _ _ _ _ _ _ _-- _ _-- _ _ _ ---------------------------
          |                   --                  -------               ---                 
          |                   -                                           --                
          |                   -                                            -                
          |                   --                                           -                
          |                    --                                         -                 
          |                      --                                      -                  
          |                       ---                                 ---                   
          |                          ----                         ----                      
          |                             --------            -------                         
          |                                     ------------                                
          |                                                                                 
          |                                                                                 
          |                                                                                 
 -2.00e+00|_________________________________________________________________________________
           -2.00e+00                            0.00e+00                            2.00e+00

>>> print(random.gauss(0, 1, [2, 3, 4]))
>>> print(random.rand([2, 3, 4]))
>>> print(random.randint(0, 9, [2, 3, 4]))
>>> print(random.uniform(0, 9, [2, 3, 4]))

[[[0.005010042633490881, 1.1160375815053902, 0.6145920379300898, -1.4696487204627253], [-0.20685462876933186, 0.8275330804972041, -0.8377832703632173, -0.8880186869697656], [-0.2653914684173608, -0.5205164919803434, -0.08359499889147641, -0.3006165927585791]], [[-1.1666695379454972, -1.0979019033440636, 0.5647293393684544, 0.23438322147503707], [0.04298318405503412, -0.6059076560822075, 1.600626179545926, 0.5204087192933082], [-0.058768641542423485, -0.4369666543837353, 0.37851158006771385, 2.0777148219436796]]]
[[[0.40140286579987816, 0.07095255870174488, 0.6446608375143889, 0.6279016180497422], [0.804158734480493, 0.38595139889111474, 0.5653398643367361, 0.9106406788835898], [0.8502113481455789, 0.5679511415517262, 0.667955293914048, 0.43668222316158123]], [[0.06619508720421818, 0.09573784118592021, 0.6821744904157657, 0.9052002792268913], [0.30333795786917084, 0.13357618895131063, 0.144258651211569, 0.648655098110358], [0.8474099644680997, 0.8461881711073397, 0.6529621910052777, 0.17709859779327897]]]
[[[1, 3, 9, 9], [0, 8, 0, 6], [5, 0, 0, 3]], [[9, 5, 6, 2], [6, 4, 9, 6], [8, 4, 8, 6]]]
[[[2.3714687054662273, 7.8682431629091605, 3.4889108978334065, 7.8710116452525885], [8.524292784475549, 6.98190581041993, 3.4297944437860264, 6.068508585966597], [5.111615446006805, 7.916996987595166, 3.589747975729174, 1.3794064763997484]], [[3.295260189867274, 5.608688777939621, 8.217536152479274, 5.209074856197099], [4.95611538157316, 3.2743034659238717, 2.7104110034788764, 2.541949514340043], [8.033753127455242, 4.943764676329522, 7.150364785741341, 6.550305532995521]]]

>>> print(regression.linear_regression(range(5), [2, 4, 6, 7, 8]))
>>> print(regression.parabolic_regression(range(5), [2, 4, 6, 7, 8]))

f(x) = 1.5 * x + 2.4
[1.5, 2.4]
f(x) = -0.214285714 * x ** 2 + 2.357142857 * x + 1.971428571
[-0.214285714, 2.357142857, 1.971428571]

>>> print(tools.classify([1, 2.3, 4 + 5j, "string", list, True, 3.14, False, tuple, tools]))
>>> print(tools.deduplicate(["Python", 6, "NumPy", int, "PyPyNum", 9, "pypynum", "NumPy", 6, True]))
>>> print(tools.frange(0, 3, 0.4))
>>> print(tools.linspace(0, 2.8, 8))

{<class 'int'>: [1], <class 'float'>: [2.3, 3.14], <class 'complex'>: [(4+5j)], <class 'str'>: ['string'], <class 'type'>: [<class 'list'>, <class 'tuple'>], <class 'bool'>: [True, False], <class 'module'>: [<module 'pypynum.tools' from 'F:\\PyPyproject\\PyPyproject1\\pypynum\\tools.py'>]}
['Python', 6, 'NumPy', <class 'int'>, 'PyPyNum', 9, 'pypynum', True]
[0.0, 0.4, 0.8, 1.2000000000000002, 1.6, 2.0, 2.4000000000000004, 2.8000000000000003]
[0.0, 0.39999999999999997, 0.7999999999999999, 1.2, 1.5999999999999999, 1.9999999999999998, 2.4, 2.8]

提示：

测试已成功通过并结束。

这些测试只是这个包功能的一部分。

更多的功能需要自己探索和尝试！

Tip:

The test has been successfully passed and ended.

These tests are only part of the functionality of this package.

More features need to be explored and tried by yourself!
```
