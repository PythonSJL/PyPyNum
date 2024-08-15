"""
Special types collection
"""
from decimal import Decimal
from typing import *

arr = Union[list, tuple]
ite = Union[list, tuple, str]
num = Union[int, float, complex]
prec = Union[int, float, str, Decimal]
real = Union[int, float]
