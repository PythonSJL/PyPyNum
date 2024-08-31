from pypynum.Array import *
from pypynum.bessel import *
from pypynum.chars import int2superscript, superscript2int, int2subscript, subscript2int
from pypynum.cipher import *
from pypynum.equations import *
from pypynum.file import *
from pypynum.FourierT import *
from pypynum.maths import *
from pypynum.multiprec import *
from pypynum.numbers import *
from pypynum.types import Any, Callable, Dict, Iterator, List, Tuple


def 数组(数据: list = None, 检查: bool = True) -> Array:
    return Array(数据, 检查)


def 转换为列表(数据) -> list:
    return aslist(数据)


def 转换为数组(数据) -> Array:
    return asarray(数据)


def 全部填充(形状, 填充值, 返回类型=Array):
    return full(形状, 填充值, 返回类型)


def 类似形状填充(数组A, 填充值, 返回类型=Array):
    return full_like(数组A, 填充值, 返回类型)


def 全零(形状, 返回类型=Array):
    return full(形状, 0, 返回类型)


def 类似形状全零(数组A, 返回类型=Array):
    return full_like(数组A, 0, 返回类型)


def 全一(形状, 返回类型=Array):
    return full(形状, 1, 返回类型)


def 类似形状全一(数组A, 返回类型=Array):
    return full_like(数组A, 1, 返回类型)


def 填充序列(形状, 序列=None, 重复=True, 填充=0, 返回类型=Array):
    return fill(形状, 序列, 重复, 填充, 返回类型)


def 整数转上标(标准字符串: str) -> str:
    return int2superscript(标准字符串)


def 上标转整数(上标字符串: str) -> str:
    return superscript2int(上标字符串)


def 整数转下标(标准字符串: str) -> str:
    return int2subscript(标准字符串)


def 下标转整数(下标字符串: str) -> str:
    return subscript2int(下标字符串)


def base64密码(文本: str, 解密: bool = False) -> str:
    return base_64(文本, 解密)


def 阿特巴什密码(文本: str) -> str:
    return atbash(文本)


def ROT13密码(文本: str) -> str:
    return rot13(文本)


def 凯撒密码(文本: str, 移位: int, 解密: bool = False) -> str:
    return caesar(文本, 移位, 解密)


def 维吉尼亚密码(文本: str, 密钥: str, 解密: bool = False) -> str:
    return vigenere(文本, 密钥, 解密)


def 代替密码(文本: str, 替换映射: dict, 解密: bool = False) -> str:
    return substitution(文本, 替换映射, 解密)


def 莫尔斯密码(文本: str, 解密: bool = False) -> str:
    return morse(文本, 解密)


def 普莱费尔密码(文本: str, 密钥: str, 解密: bool = False) -> str:
    return playfair(文本, 密钥, 解密)


def 希尔256密码(文本: bytes, 密钥: list, 解密: bool = False) -> bytes:
    return hill256(文本, 密钥, 解密)


def RC4初始化密钥调度算法(密钥: bytes) -> list:
    return ksa(密钥)


def RC4伪随机生成算法(密钥序列: list):
    return prga(密钥序列)


def RC4密码(文本: bytes, 密钥: bytes) -> bytes:
    return rc4(文本, 密钥)


def 线性方程组(左边: list, 右边: list) -> list:
    return lin_eq(左边, 右边)


def 多项式方程(系数: list) -> list:
    return poly_eq(系数)


def 读取(文件: str) -> list:
    return read(文件)


def 写入(文件: str, *对象: object):
    return write(文件, *对象)


def 一维傅里叶变换(*数据) -> FT1D:
    return FT1D(*数据)


def Fraction转为Decimal(分数对象: Fraction, 有效位数: int) -> Decimal:
    return frac2dec(分数对象, 有效位数)


def 多精度欧拉数(有效位数: int, 方法: str = "series") -> Decimal:
    return mp_e(有效位数, 方法)


def 多精度圆周率(有效位数: int, 方法: str = "chudnovsky") -> Decimal:
    return mp_pi(有效位数, 方法)


def 多精度黄金分割率(有效位数: int, 方法: str = "algebraic") -> Decimal:
    return mp_phi(有效位数, 方法)


def 多精度正弦(x: real, 有效位数: int) -> Decimal:
    return mp_sin(x, 有效位数)


def 多精度余弦(x: real, 有效位数: int) -> Decimal:
    return mp_cos(x, 有效位数)


def 多精度自然对数(真数: real, 有效位数: int, 使用内置方法: bool = True) -> Decimal:
    return mp_ln(真数, 有效位数, 使用内置方法)


def 多精度对数(真数: real, 底数: real, 有效位数: int, 使用内置方法: bool = True) -> Decimal:
    return mp_log(真数, 底数, 有效位数, 使用内置方法)


def 多精度反正切(x: real, 有效位数: int) -> Decimal:
    return mp_atan(x, 有效位数)


def 多精度方位角(y: real, x: real, 有效位数: int) -> Decimal:
    return mp_atan2(y, x, 有效位数)


def 多精度自然指数(指数: real, 有效位数: int, 使用内置方法: bool = True) -> Decimal:
    return mp_exp(指数, 有效位数, 使用内置方法)


def 多精度双曲正弦(x: real, 有效位数: int) -> Decimal:
    return mp_sinh(x, 有效位数)


def 多精度双曲余弦(x: real, 有效位数: int) -> Decimal:
    return mp_cosh(x, 有效位数)


def 多精度反正弦(x: real, 有效位数: int) -> Decimal:
    return mp_asin(x, 有效位数)


def 多精度反余弦(x: real, 有效位数: int) -> Decimal:
    return mp_acos(x, 有效位数)


def 多精度菲涅耳正弦积分(x: real, 有效位数: int) -> Decimal:
    return mp_fresnel_s(x, 有效位数)


def 多精度菲涅耳余弦积分(x: real, 有效位数: int) -> Decimal:
    return mp_fresnel_c(x, 有效位数)


def 多精度复数(实部: prec, 虚部: prec, 有效位数: int = 28) -> MPComplex:
    return MPComplex(实部, 虚部, 有效位数)


def 转为多精度复数(实部: Union[int, float, str, Decimal, complex, MPComplex], 虚部: prec,
                   有效位数: int = 28) -> MPComplex:
    return asmpc(实部, 虚部, 有效位数)


def 下伽玛(s: num, x: num) -> num:
    return lower_gamma(s, x)


def 上伽玛(s: num, x: num) -> num:
    return upper_gamma(s, x)


def 贝塞尔函数J0(x: num) -> num:
    return bessel_j0(x)


def 贝塞尔函数J1(x: num) -> num:
    return bessel_j1(x)


def 贝塞尔函数Jv(v: real, x: num) -> num:
    return bessel_jv(v, x)


def 贝塞尔函数I0(x: num) -> num:
    return bessel_i0(x)


def 贝塞尔函数I1(x: num) -> num:
    return bessel_i1(x)


def 贝塞尔函数Iv(v: real, x: num) -> num:
    return bessel_iv(v, x)


def 乘积和(*数组: List[Any]) -> float:
    return sumprod(*数组)


def x对数y乘积(x: float, y: float) -> float:
    return xlogy(x, y)


def 序列滚动(序列: Iterator[Any], 偏移: int) -> Iterator[Any]:
    return roll(序列, 偏移)


def y次方根(被开方数: num, 开方数: num) -> num:
    return root(被开方数, 开方数)


def 自然指数(指数: real) -> real:
    return exp(指数)


def 自然对数(真数: real) -> real:
    return ln(真数)


def 最大公约数(*args: int) -> int:
    return gcd(*args)


def 最小公倍数(*args: int) -> int:
    return lcm(*args)


def 正弦(x: real) -> real:
    return sin(x)


def 余弦(x: real) -> real:
    return cos(x)


def 正切(x: real) -> real:
    return tan(x)


def 余割(x: real) -> real:
    return csc(x)


def 正割(x: real) -> real:
    return sec(x)


def 余切(x: real) -> real:
    return cot(x)


def 反正弦(x: real) -> real:
    return asin(x)


def 反余弦(x: real) -> real:
    return acos(x)


def 反正切(x: real) -> real:
    return atan(x)


def 反余割(x: real) -> real:
    return acsc(x)


def 反正割(x: real) -> real:
    return asec(x)


def 反余切(x: real) -> real:
    return acot(x)


def 双曲正弦(x: real) -> real:
    return sinh(x)


def 双曲余弦(x: real) -> real:
    return cosh(x)


def 双曲正切(x: real) -> real:
    return tanh(x)


def 双曲余割(x: real) -> real:
    return csch(x)


def 双曲正割(x: real) -> real:
    return sech(x)


def 双曲余切(x: real) -> real:
    return coth(x)


def 反双曲正弦(x: real) -> real:
    return asinh(x)


def 反双曲余弦(x: real) -> real:
    return acosh(x)


def 反双曲正切(x: real) -> real:
    return atanh(x)


def 反双曲余割(x: real) -> real:
    return acsch(x)


def 反双曲正割(x: real) -> real:
    return asech(x)


def 反双曲余切(x: real) -> real:
    return acoth(x)


def 极差(数据: List[float]) -> float:
    return ptp(数据)


def 中位数(数据: List[float]) -> float:
    return median(数据)


def 频率统计(数据: List[Any]) -> Dict[Any, int]:
    return freq(数据)


def 众数(数据: List[Any]):
    return mode(数据)


def 平均数(数据: List[float]) -> float:
    return mean(数据)


def 几何平均数(数据: List[float]) -> float:
    return geom_mean(数据)


def 平方平均数(数据: List[float]) -> float:
    return square_mean(数据)


def 调和平均数(数据: List[float]) -> float:
    return harm_mean(数据)


def 原点矩(数据: List[float], 阶数: int) -> float:
    return raw_moment(数据, 阶数)


def 中心矩(数据: List[float], 阶数: int) -> float:
    return central_moment(数据, 阶数)


def 方差(数据: List[float], 自由度: int = 0) -> float:
    return var(数据, 自由度)


def 偏度(数据: List[float]) -> float:
    return skew(数据)


def 峰度(数据: List[float], 费希尔: bool = True) -> float:
    return kurt(数据, 费希尔)


def 标准差(数据: List[float], 自由度: int = 0) -> float:
    return std(数据, 自由度)


def 协方差(x: List[float], y: List[float], 自由度: int = 0) -> float:
    return cov(x, y, 自由度)


def 相关系数(x: List[float], y: List[float]) -> float:
    return corr_coeff(x, y)


def 判定系数(x: List[float], y: List[float]) -> float:
    return coeff_det(x, y)


def 分位数(数据: list, 分位值: float, 插值方法: str = "linear", 已排序: bool = False) -> float:
    return quantile(数据, 分位值, interpolation=插值方法, ordered=已排序)


def 积累乘积(数据: List[float]) -> float:
    return product(数据)


def 连续加和(下界: int, 上界: int, 函数: Callable) -> float:
    return sigma(下界, 上界, 函数)


def 连续乘积(下界: int, 上界: int, 函数: Callable) -> float:
    return pi(下界, 上界, 函数)


def 导数(函数, 参数: float, 步长: float = 1e-6, *额外参数, **额外关键字参数) -> float:
    return deriv(函数, 参数, 步长, *额外参数, **额外关键字参数)


def 积分(函数, 积分开始: float, 积分结束: float, 积分点数: int = 1000000, *额外参数, **额外关键字参数) -> float:
    return integ(函数, 积分开始, 积分结束, 积分点数, *额外参数, **额外关键字参数)


def 贝塔函数(p: float, q: float) -> float:
    return beta(p, q)


def 伽玛函数(alpha: float) -> float:
    return gamma(alpha)


def 阶乘函数(n: int) -> int:
    return factorial(n)


def 排列数(总数: int, 选取数: int) -> int:
    return arrangement(总数, 选取数)


def 组合数(总数: int, 选取数: int) -> int:
    return combination(总数, 选取数)


def 黎曼函数(alpha: float) -> float:
    return zeta(alpha)


def 误差函数(x: real) -> real:
    return erf(x)


def S型函数(x: real) -> real:
    return sigmoid(x)


def 符号函数(x: num) -> num:
    return sign(x)


def 负一整数次幂(指数: int) -> int:
    return parity(指数)


def 累加和(序列: List[float]) -> List[float]:
    return cumsum(序列)


def 累乘积(序列: List[float]) -> List[float]:
    return cumprod(序列)


def 多次方根取整(被开方数: int, 开方数: int) -> int:
    return iroot(被开方数, 开方数)


def 欧拉函数(n: int) -> int:
    return totient(n)


def 模运算阶(a: int, n: int, b: int) -> int:
    return mod_order(a, n, b)


def 原根(a: int, 单个: bool = False) -> Union[int, List[int]]:
    return primitive_root(a, 单个)


def 归一化(数据: List[float], 目标: float = 1) -> List[float]:
    return normalize(数据, 目标)


def 加权平均(数据: List[float], 权重: List[float]) -> float:
    return average(数据, 权重)


def 扩展欧几里得算法(a: int, b: int) -> Tuple[int, int, int]:
    return exgcd(a, b)


def 中国剩余定理(n: List[int], a: List[int]) -> int:
    return crt(n, a)


def 平方根取整(被开方数: int) -> int:
    return isqrt(被开方数)


def 可能是平方数(n: int) -> bool:
    return is_possibly_square(n)


def 判断平方数(n: int) -> bool:
    return is_square(n)


def 整数转单词(整数: int) -> str:
    return int2words(整数)


def 字符串转整数(字符串: str) -> int:
    return str2int(字符串)


def 整数转罗马数(整数: int, 上划线: bool = True) -> str:
    return int2roman(整数, 上划线)


def 罗马数转整数(罗马数: str) -> int:
    return roman2int(罗马数)


def 浮点数转分数(数值: float, 是否带分数: bool = False, 误差: float = 1e-15) -> tuple:
    return float2fraction(数值, 是否带分数, 误差)


def 拆分浮点数字符串(字符串: str) -> tuple:
    return split_float(字符串)


def 解析浮点数字符串(字符串: str) -> tuple:
    return parse_float(字符串)
