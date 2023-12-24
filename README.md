# PyPyNum

## Version -> 1.0.0
### Gitee -> https://www.gitee.com/PythonSJL/PyPyNum

#### 介绍
#### Introduction
1.  DIY数学库，类似于numpy、scipy等，专为PyPy解释器制作
1.  DIY math library, similar to numpy, scipy, etc., specifically designed for PyPy interpreters
2.  不定期更新版本，增加更多实用功能
2.  Update versions periodically to add more practical features
3.  如需联系，QQ 2261748025 （Py𝙿𝚢𝚝𝚑𝚘𝚗-水晶兰）
3.  If you need to contact, QQ 2261748025 (Py𝙿𝚢𝚝𝚑𝚘𝚗-水晶兰)

#### 基本结构
#### Basic structure
```
PyPyNum
    test
        [A Code Test File]
    Geometry
        CLASSES
            Circle
            Line
            Point
            Polygon
            Quadrilateral
            Triangle
        FUNCTIONS
            distance(g1: <built-in function any>, g2: <built-in function any>, error: int | float = 0) -> float
    Matrix
        CLASSES
            Matrix
        FUNCTIONS
            eig(matrix)
            hypot(a, b)
            identity(n)
            lu(matrix)
            mat(data)
            qr(matrix)
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
    Tensor
        CLASSES
            Tensor
        FUNCTIONS
            get_shape(_tensor)
            is_valid_tensor(_tensor, _shape)
            ten(data)
            tensor_and_number(tensor, operator, number)
            tolist(_nested_list)
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
    plotting
        FUNCTIONS
            background(right: int | float = 5, left: int | float = -5, top: int | float = 5, bottom: int | float = -5, complexity: int | float = 5, ratio: int | float = 3, merge: bool = False) -> list | str
            binary(function, right: int | float = 5, left: int | float = -5, top: int | float = 5, bottom: int | float = -5, complexity: int | float = 5, ratio: int | float = 3, error=0, compare='==', merge: bool = True, basic: list = None, character: str = '.', data: bool = False) -> list | str
            c_unary(function, start: int | float, end: int | float, interval: int | float = 5, projection: str = 'ri', right: int | float = 5, left: int | float = -5, top: int | float = 5, bottom: int | float = -5, complexity: int | float = 5, ratio: int | float = 3, merge: bool = True, basic: list = None, character: str = '.', data: bool = False) -> list | str
            change(data: list | str) -> list | str
            unary(function, right: int | float = 5, left: int | float = -5, top: int | float = 5, bottom: int | float = -5, complexity: int | float = 5, ratio: int | float = 3, merge: bool = True, basic: list = None, character: str = '.', data: bool = False) -> list | str
    regression
        FUNCTIONS
            __covariance_matrix(_x, _y)
            linear_regression(x_values, y_values)
            parabolic_regression(x, y)
            __solve_equations(a1, b1, c1, a2, b2, c2)
    tools
        FUNCTIONS
            classify(array: list | tuple) -> dict
            deduplicate(iterable: list | tuple | str) -> list | tuple | str
            frange(start: int | float, stop: int | float, step: float = 1.0) -> list
            linspace(start: int | float, stop: int | float, number: int) -> list
```

#### 代码测试
#### Code testing
```

>>> from pypynum import Geometry, Matrix, Quaternion, Tensor, constants, equations, mathematics, regression, plotting, tools

...

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

>>> print(regression.linear_regression(range(5), [2, 4, 6, 7, 8]))
>>> print(regression.parabolic_regression(range(5), [2, 4, 6, 7, 8]))

f(x) = 1.5 * x + 2.4
[1.5, 2.4]
f(x) = -0.214285714 * x ** 2 + 2.357142857 * x + 1.971428571
[-0.214285714, 2.357142857, 1.971428571]

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

>>> print(tools.classify([1, 2.3, 4 + 5j, "string", list, True, 3.14, False, tuple, tools]))
>>> print(tools.deduplicate(["Python", 6, "NumPy", int, "PyPyNum", 9, "pypynum", "NumPy", 6, True]))
>>> print(tools.frange(0, 3, 0.4))
>>> print(tools.linspace(0, 2.8, 8))

{<class 'int'>: [1], <class 'float'>: [2.3, 3.14], <class 'complex'>: [(4+5j)], <class 'str'>: ['string'], <class 'type'>: [<class 'list'>, <class 'tuple'>], <class 'bool'>: [True, False], <class 'module'>: [<module 'pypynum.tools' from 'F:\\PyPyproject\\PyPyproject1\\pypynum\\tools.py'>]}
['Python', 6, 'NumPy', <class 'int'>, 'PyPyNum', 9, 'pypynum', True]
[0.0, 0.4, 0.8, 1.2000000000000002, 1.6, 2.0, 2.4000000000000004, 2.8000000000000003]
[0.0, 0.39999999999999997, 0.7999999999999999, 1.2, 1.5999999999999999, 1.9999999999999998, 2.4, 2.8]
```

#### 安装教程
#### Installation Tutorial
1.  下载即可使用，无需其他依赖项
1.  Download and use without any other dependencies

#### 参与贡献
#### Participate in contributions
1.  Fork 本仓库
1.  Fork warehouse
2.  新建 Feat_xxx 分支
2.  Create a new Feature_ Xxx branch
3.  提交代码
3.  Submit code
4.  新建 Pull Request
4.  Create a new Pull Request

#### 特技
#### Stunts
1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
1.  Use Readme\_ XXX. MD to support different languages, such as Readme\_ En.md, Readme\_ Zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
2.  Gitee Official Blog [blog. Gitee. com]（ https://blog.gitee.com ）
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
3.  You can[ https://gitee.com/explore ]（ https://gitee.com/explore ）This address is for learning about excellent open source projects on Gitee
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
4.  [GVP]（ https://gitee.com/gvp ）The full name is Gitee's most valuable open source project, which is comprehensively evaluated as an excellent open source project
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
5.  Gitee Official User Manual[ https://gitee.com/help ]（ https://gitee.com/help ）
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
6.  The cover character of Gitee is a column used to showcase the style of Gitee members[ https://gitee.com/gitee-stars/ ]（ https://gitee.com/gitee-stars/ ）