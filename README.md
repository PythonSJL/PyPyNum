# <font color = blue>PyPyNum</font>

<font color = gree>A Python math package written in pure Python programming language</font><font color = red>
(python_requires >= 3.5)</font>

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

## Version -> 1.6.0 | PyPI -> https://pypi.org/project/PyPyNum/ | Gitee -> https://www.gitee.com/PythonSJL/PyPyNum

![logo](pypynum/PyPyNum.png)

PyPI上无法显示logo，可以在Gitee中查看。

The logo cannot be displayed on PyPI, it can be viewed in Gitee.

### 介绍

#### Introduction

+ DIY数学库，类似于numpy、scipy等，专为PyPy解释器制作
+ DIY math library, similar to numpy, scipy, etc., specifically designed for PyPy interpreters
+ 不定期更新版本，增加更多实用功能
+ Update versions periodically to add more practical features
+ 如需联系，QQ 2261748025 （Py𝙿𝚢𝚝𝚑𝚘𝚗-水晶兰）
+ If you need to contact, QQ 2261748025 (Py𝙿𝚢𝚝𝚑𝚘𝚗-水晶兰)

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

... (Do you want to view all the content?
Enter "from pypynum import this" in your
Python interpreter and run it!)

                                        February 27, 2024
```

### 与上一个版本相比新增功能

#### New features compared to the previous version

```
chars [Special mathematical symbols]
    DATA
        arrow = [["↖", "↑", "↗"], ["←", "⇌", "→"], ["↙", "↓", "↘"], ["↔", "⇋",...
        div = "÷"
        mul = "×"
        others = "¬°‰‱′″∀∂∃∅∆∇∈∉∏∐∑∝∞∟∠∣∥∧∨∩∪∫∬∭∮∯∰∴∵∷∽≈≌≒≠≡≢≤≥≪≫≮≯≰≱≲≳⊕⊙⊥⊿⌒㏑㏒...
        overline = "̄"
        pi = "Ππ𝜫𝝅𝝥𝝿𝞟𝞹Пп∏ϖ∐ℼㄇ兀"
        sgn = "±"
        strikethrough = "̶"
        subscript = "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"
        superscript = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛ...
        tab = [["┌", "┬", "┐"], ["├", "┼", "┤"], ["└", "┴", "┘"], ["─", "╭", "...
        underline = "_"

maths [Mathematical functions]
    ...
    raw_moment(data: Union[list, tuple], order: int) -> float
    central_moment(data: Union[list, tuple], order: int) -> float
    skew(data: Union[list, tuple]) -> float
    kurt(data: Union[list, tuple]) -> float
    ...

numbers [Conversion of various numbers]
    FUNCTIONS
        float2fraction(number: float, mixed: bool = False, error: float = 1e-15) -> tuple
        int2roman(integer: int, overline: bool = True) -> str
        roman2int(roman_num: str) -> int
        str2int(string: str) -> int
    DATA
        roman_symbols = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
        roman_values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
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
    ★ __init__
        [Import some features from other modules in this package]
    ★ errors [Special errors]
        CLASSES
            LogicError
            RandomError
            ShapeError
    ★ file [Reading and saving instance data]
        FUNCTIONS
            read(file: str) -> list
            write(file: str, *cls: object)
    ★ test
        [A code test file]
    ★ this
        [The Zen of PyPyNum]
    ★ types [Special types]
        DATA
            arr = typing.Union[list, tuple]
            ite = typing.Union[list, tuple, str]
            num = typing.Union[int, float, complex]
            real = typing.Union[int, float]
    ★ Array [N-dimensional array]
        CLASSES
            Array
        FUNCTIONS
            array(data=None)
            fill(shape, sequence=None)
            function(_array, _function, args=None)
            get_shape(data)
            is_valid_array(_array, _shape)
            zeros(shape)
            zeros_like(_nested_list)
    ★ FourierT [Fourier transform and inverse Fourier transform]
        CLASSES
            FT1D
    ★ Geometry [Planar geometry]
        CLASSES
            Circle
            Line
            Point
            Polygon
            Quadrilateral
            Triangle
        FUNCTIONS
            distance(g1, g2, error: int | float = 0) -> float
    ★ Group [Group theory]
        CLASSES
            Group
        FUNCTIONS
            add(x, y)
            divide(x, y)
            group(data)
            multiply(x, y)
            subtract(x, y)
    ★ Logic [Logic circuit simulation]
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
    ★ Matrix [Matrix calculation]
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
    ★ NeuralN [A simple neural network model]
        CLASSES
            NeuralNetwork
        FUNCTIONS
            neuraln(_input, _hidden, _output)
    ★ Quaternion [Quaternion calculation]
        CLASSES
            Euler
            Quaternion
        FUNCTIONS
            change(data: Union[pypynum.Quaternion.Quaternion, pypynum.Matrix.Matrix, pypynum.Quaternion.Euler], to: str) -> Union[pypynum.Quaternion.Quaternion, pypynum.Matrix.Matrix, pypynum.Quaternion.Euler]
            euler(yaw: Union[int, float] = 0, pitch: Union[int, float] = 0, roll: Union[int, float] = 0) -> pypynum.Quaternion.Euler
            quat(w: Union[int, float] = 0, x: Union[int, float] = 0, y: Union[int, float] = 0, z: Union[int, float] = 0) -> pypynum.Quaternion.Quaternion
    ★ Symbolics [Symbol calculation]
        FUNCTIONS
            interpreter(expr: str) -> list
        DATA
            basic = "%()*+-./0123456789"
            english = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            greek = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω"
            operators = ["**", "*", "//", "/", "%", "+", "-"]
            valid = "%()*+-./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcd...zΑΒΓΔΕΖΗΘΙ...
    ★ Tensor [Tensor calculation]
        CLASSES
            Tensor
        FUNCTIONS
            ten(data)
            tensor_and_number(tensor, operator, number)
            tolist(_nested_list)
            zeros(_dimensions)
            zeros_like(_nested_list)
    ★ Vector [Vector calculation]
        CLASSES
            Vector
        FUNCTIONS
            same(length, value=0)
            vec(data)
            zeros(_dimensions)
            zeros_like(_nested_list)
    ★ chars [Special mathematical symbols]
        DATA
            arrow = [["↖", "↑", "↗"], ["←", "⇌", "→"], ["↙", "↓", "↘"], ["↔", "⇋",...
            div = "÷"
            mul = "×"
            others = "¬°‰‱′″∀∂∃∅∆∇∈∉∏∐∑∝∞∟∠∣∥∧∨∩∪∫∬∭∮∯∰∴∵∷∽≈≌≒≠≡≢≤≥≪≫≮≯≰≱≲≳⊕⊙⊥⊿⌒㏑㏒...
            overline = "̄"
            pi = "Ππ𝜫𝝅𝝥𝝿𝞟𝞹Пп∏ϖ∐ℼㄇ兀"
            sgn = "±"
            strikethrough = "̶"
            subscript = "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"
            superscript = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛ...
            tab = [["┌", "┬", "┐"], ["├", "┼", "┤"], ["└", "┴", "┘"], ["─", "╭", "...
            underline = "_"
    ★ cipher [String encryption and decryption algorithms]
        FUNCTIONS
            dna(string: str, decrypt: bool = False) -> str
    ★ constants [Constants in mathematics and science]
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
            deka = 10
            e = 2.718281828459045
            exa = 1000000000000000000
            femto = 1e-15
            gamma = 0.5772156649015329
            giga = 1000000000
            h = 6.62607015e-34
            hecto = 100
            inf = inf
            kilo = 1000
            mega = 1000000
            micro = 1e-06
            milli = 0.001
            nan = nan
            nano = 1e-09
            peta = 1000000000000000
            phi = 1.618033988749895
            pi = 3.141592653589793
            pico = 1e-12
            qe = 1.60217733e-19
            tera = 1000000000000
            yocto = 1e-24
            yotta = 1000000000000000000000000
            zepto = 1e-21
            zetta = 1000000000000000000000
    ★ equations [Solving specific forms of equations]
        FUNCTIONS
            linear_equation(left: list, right: list) -> list
            polynomial_equation(coefficients: list) -> list
    ★ maths [Mathematical functions]
        FUNCTIONS
            A = arrangement(n: int, r: int) -> int
            C = combination(n: int, r: int) -> int
            acos(x: Union[int, float]) -> Union[int, float]
            acosh(x: Union[int, float]) -> Union[int, float]
            acot(x: Union[int, float]) -> Union[int, float]
            acoth(x: Union[int, float]) -> Union[int, float]
            acsc(x: Union[int, float]) -> Union[int, float]
            acsch(x: Union[int, float]) -> Union[int, float]
            arrangement(n: int, r: int) -> int
            asec(x: Union[int, float]) -> Union[int, float]
            asech(x: Union[int, float]) -> Union[int, float]
            asin(x: Union[int, float]) -> Union[int, float]
            asinh(x: Union[int, float]) -> Union[int, float]
            atan(x: Union[int, float]) -> Union[int, float]
            atanh(x: Union[int, float]) -> Union[int, float]
            beta(p: Union[int, float], q: Union[int, float]) -> Union[int, float]
            central_moment(data: Union[list, tuple], order: int) -> float
            coeff_det(x: Union[list, tuple], y: Union[list, tuple]) -> Union[int, float, complex]
            combination(n: int, r: int) -> int
            corr_coeff(x: Union[list, tuple], y: Union[list, tuple]) -> Union[int, float, complex]
            cos(x: Union[int, float]) -> Union[int, float]
            cosh(x: Union[int, float]) -> Union[int, float]
            cot(x: Union[int, float]) -> Union[int, float]
            coth(x: Union[int, float]) -> Union[int, float]
            cov(x: Union[list, tuple], y: Union[list, tuple]) -> Union[int, float, complex]
            csc(x: Union[int, float]) -> Union[int, float]
            csch(x: Union[int, float]) -> Union[int, float]
            cumprod(lst: Union[list, tuple]) -> list
            cumsum(lst: Union[list, tuple]) -> list
            definite_integral(f, x_start: Union[int, float], x_end: Union[int, float], n: int = 10000000) -> float
            derivative(f, x: Union[int, float], h: Union[int, float] = 1e-07) -> float
            erf(x: Union[int, float]) -> float
            exp(x: Union[int, float]) -> Union[int, float]
            factorial(n: int) -> int
            freq(data: Union[list, tuple]) -> dict
            gamma(alpha: Union[int, float]) -> float
            gaussian(x: Union[int, float], _mu: Union[int, float] = 0, _sigma: Union[int, float] = 1) -> float
            gcd(*args: int) -> int
            geom_mean(numbers: Union[list, tuple]) -> Union[int, float, complex]
            harm_mean(numbers: Union[list, tuple]) -> Union[int, float, complex]
            iroot(y: int, n: int) -> int
            kurt(data: Union[list, tuple]) -> float
            lcm(*args: int) -> int
            ln(x: Union[int, float]) -> Union[int, float]
            mean(numbers: Union[list, tuple]) -> Union[int, float, complex]
            median(numbers: Union[list, tuple]) -> Union[int, float, complex]
            parity(x: int) -> int
            pi(i: int, n: int, f) -> Union[int, float, complex]
            poisson(x: int, _lambda: Union[int, float]) -> float
            product(numbers: Union[list, tuple]) -> Union[int, float, complex]
            ptp(numbers: Union[list, tuple]) -> Union[int, float, complex]
            raw_moment(data: Union[list, tuple], order: int) -> float
            root(x: Union[int, float, complex], y: Union[int, float, complex]) -> Union[int, float, complex]
            sec(x: Union[int, float]) -> Union[int, float]
            sech(x: Union[int, float]) -> Union[int, float]
            sigma(i: int, n: int, f) -> Union[int, float, complex]
            sigmoid(x: Union[int, float]) -> float
            sign(x: Union[int, float]) -> int
            sin(x: Union[int, float]) -> Union[int, float]
            sinh(x: Union[int, float]) -> Union[int, float]
            skew(data: Union[list, tuple]) -> float
            square_mean(numbers: Union[list, tuple]) -> Union[int, float, complex]
            std(numbers: Union[list, tuple]) -> Union[int, float, complex]
            tan(x: Union[int, float]) -> Union[int, float]
            tanh(x: Union[int, float]) -> Union[int, float]
            var(numbers: Union[list, tuple]) -> Union[int, float, complex]
            zeta(alpha: Union[int, float]) -> float
    ★ numbers [Conversion of various numbers]
        FUNCTIONS
            float2fraction(number: float, mixed: bool = False, error: float = 1e-15) -> tuple
            int2roman(integer: int, overline: bool = True) -> str
            roman2int(roman_num: str) -> int
            str2int(string: str) -> int
        DATA
            roman_symbols = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
            roman_values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    ★ plotting [Draw a graph of equations using characters]
        FUNCTIONS
            color(text: str, rgb: arr) -> str
            change(data: thing) -> thing
            background(right: real = 5, left: real = -5, top: real = 5, bottom: real = -5...
            unary(function, right: real = 5, left: real = -5, top: real = 5, bottom: real = -5, complexity: real = 5...
            binary(function, right: real = 5, left: real = -5, top: real = 5, bottom: real = -5, complexity: real = 5...
            c_unary(function, start: real, end: real, interval: real = 5, projection: str = "ri", right: real = 5...
    ★ probability [Probability function]
        FUNCTIONS
            binomial(sample_size: int, successes: int, success_probability: Union[int, float]) -> float
            hypergeometric(total_items: int, success_items: int, sample_size: int, successes_in_sample: int) -> float
    ★ random [Generate random numbers or random arrays]
        FUNCTIONS
            choice(seq: Union[list, tuple, str], shape: Union[list, tuple] = None)
            gauss(mu: Union[int, float] = 0, sigma: Union[int, float] = 1, shape: Union[list, tuple] = None) -> Union[float, list]
            gauss_error(original: Union[list, tuple], mu: Union[int, float] = 0, sigma: Union[int, float] = 1) -> list
            rand(shape: Union[list, tuple] = None) -> Union[float, list]
            randint(a: int, b: int, shape: Union[list, tuple] = None) -> Union[int, list]
            uniform(a: Union[int, float], b: Union[int, float], shape: Union[list, tuple] = None) -> Union[float, list]
    ★ regression [Formula based polynomial regression]
        FUNCTIONS
            linear_regression(x: Union[list, tuple], y: Union[list, tuple]) -> list
            parabolic_regression(x: Union[list, tuple], y: Union[list, tuple]) -> list
            polynomial_regression(x: Union[list, tuple], y: Union[list, tuple], n: int = None) -> list
    ★ sequence [Various sequences]
        FUNCTIONS
            fibonacci(n: int, single: bool = True) -> Union[int, list]
            catalan(n: int, single: bool = True) -> Union[int, list]
            bernoulli(n: int, single: bool = True) -> list
            recaman(n: int, single: bool = True) -> Union[int, list]
            arithmetic_sequence(*, a1: real = None, an: real = None, d: real = None, n: real = None, s: real = None) -> dict
            geometric_sequence(*, a1: real = None, an: real = None, r: real = None, n: real = None, s: real = None) -> dict
    ★ tools [Other useful tools]
        FUNCTIONS
            classify(array: Union[list, tuple]) -> dict
            deduplicate(iterable: Union[list, tuple, str]) -> Union[list, tuple, str]
            frange(start: Union[int, float], stop: Union[int, float], step: float = 1.0) -> list
            linspace(start: Union[int, float], stop: Union[int, float], number: int) -> list
            split(iterable: Union[list, tuple, str], key: Union[list, tuple], retain: bool = False) -> list
```

### 代码测试

#### Code testing

```
>>> from pypynum import (Array, Geometry, Logic, Matrix, Quaternion, Symbolics, Tensor, Vector,
                         cipher, constants, equations, maths, plotting, random, regression, tools)

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

>>> a, b, c = 1, 1, 1
>>> adder0, adder1 = Logic.HalfAdder("alpha", a, b), Logic.HalfAdder("beta", c, None)
>>> xor0 = Logic.XOR("alpha")
>>> ff0, ff1 = Logic.DFF("alpha"), Logic.DFF("beta")
>>> xor0.set_order0(1)
>>> xor0.set_order1(1)
>>> Logic.connector(adder0, adder1)
>>> Logic.connector(adder0, xor0)
>>> Logic.connector(adder1, xor0)
>>> Logic.connector(adder1, ff0)
>>> Logic.connector(xor0, ff1)
>>> print("sum: {}, carry: {}".format(ff0.out(), ff1.out()))

sum: [1], carry: [1]

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
ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω
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
>>> print(equations.polynomial_equation(p))
>>> print(equations.linear_equation(*m))

[(-1.5615528128088307-6.5209667308287455e-24j)   (2.5615528128088294+4.456233626665941e-24j)   (1.0000000000000007+3.241554513744382e-25j)]
[ 1.6666666666666667 -0.6666666666666666 -0.4444444444444444]

>>> print(maths.cot(constants.pi / 3))
>>> print(maths.gamma(1.5))
>>> print(maths.pi(1, 10, lambda x: x ** 2))
>>> print(maths.product([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))
>>> print(maths.sigma(1, 10, lambda x: x ** 2))
>>> print(maths.var([2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))

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

[[[0.4334341920363395, 0.055711784711422116, -1.0235500373980284, 0.30031229336738374], [-0.2650367914670356, 0.5513398538865067, -0.9735921328831166, 0.41345578602104827], [-0.11598957920080871, -0.9044539791933183, 1.6448227575237069, -0.26304156924843813]], [[0.27363898507271256, -0.5897181011789576, 1.5120937498473583, 2.1302709742844694], [1.9743293887616236, 0.4117207260898469, 0.5809554193110543, -1.8456249006764007], [1.274481044612177, -0.30645083457981553, -1.3285606156236818, 0.33473439037886943]]]
[[[0.5269441534226782, 0.36498666932667356, 0.7363066388832684, 0.5878544826035406], [0.5684721009896431, 0.9009577979323332, 0.036288112799501615, 0.18351641818419884], [0.24258369409385339, 0.09354340906140202, 0.4856203412285762, 0.783031677244552]], [[0.8777465681935882, 0.6406910705155251, 0.10275292827025073, 0.01295823682977526], [0.3898500974345528, 0.6216248983423127, 0.3179425906177036, 0.012870877167621808], [0.2660481991211192, 0.09872041627158801, 0.3681944568198672, 0.494087114885137]]]
[[[5, 9, 5, 6], [6, 7, 6, 1], [1, 3, 2, 4]], [[5, 8, 8, 3], [3, 2, 3, 9], [3, 0, 7, 1]]]
[[[8.610851610963957, 1.3747433091161905, 1.3831050577679438, 4.715182178697273], [0.8765517657148284, 4.809554825684029, 2.7557819856736137, 5.938765584746821], [6.088739464744903, 4.627722536295625, 0.6116370455995369, 5.875683438664389]], [[7.7228845997304845, 5.428461366109726, 8.02712172516869, 5.9319006090345425], [5.726626482636939, 7.978329508380601, 1.114307478513796, 6.236721706167868], [1.4123245528031072, 5.327811122183013, 7.324213082306745, 1.5016363011868927]]]

>>> print(regression.linear_regression(list(range(5)), [2, 4, 6, 7, 8]))
>>> print(regression.parabolic_regression(list(range(5)), [2, 4, 6, 7, 8]))
>>> print(regression.polynomial_regression(list(range(5)), [2, 4, 6, 7, 8], 4))

[1.5, 2.4000000000000004]
[-0.21428571428571183, 2.3571428571428474, 1.9714285714285764]
[0.08333333334800574, -0.6666666668092494, 1.4166666678382942, 1.1666666648311956, 2.0000000002900613]

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
