ContentError = ValueError("The content of the string is invalid")
ROMAN_VALUES = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
ROMAN_SYMBOLS = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]


def int2words(integer: int) -> str:
    """
    introduction
    ==========
    Convert integers into natural language.

    example
    ==========
    >>> int2words(4294967296)
    'four billion two hundred and ninety-four million nine hundred and sixty-seven thousand two hundred and ninety-six'
    >>>
    :param integer: integer
    :return:
    """
    if not isinstance(integer, int):
        raise TypeError("The input must be an integer")
    if integer == 0:
        return "zero"
    if integer < 0:
        return "negative " + int2words(abs(integer))
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    def two_digit(num):
        if num < 20:
            return ones[num] if num > 0 else ""
        ten, one = divmod(num, 10)
        return tens[ten] + ("-" if one else "") + ones[one]

    def three_digit(num):
        if num == 0:
            return ""
        hundred, rem = divmod(num, 100)
        return ones[hundred] + " hundred" + (" and " if rem else "") + two_digit(rem)

    big_units = ["quadragintillion", "novemtrigintillion", "octrigintillion", "septrigintillion", "sextrigintillion",
                 "quintrigintillion", "quattuortrigintillion", "trestrigintillion", "duotrigintillion",
                 "untrigintillion", "trigintillion", "novemvigintillion", "octovigintillion", "septenvigintillion",
                 "sexvigintillion", "quinvigintillion", "quattuorvigintillion", "trevigintillion", "duovigintillion",
                 "unvigintillion", "vigintillion", "novemdecillion", "octodecillion", "septendecillion", "sexdecillion",
                 "quindecillion", "quattuordecillion", "tredecillion", "duodecillion", "undecillion", "decillion",
                 "nonillion", "octillion", "septillion", "sextillion", "quintillion", "quadrillion", "trillion",
                 "billion", "million", "thousand"]
    unit_value = 10 ** (3 * len(big_units))
    result = []
    for name in big_units:
        if integer >= unit_value:
            value, integer = divmod(integer, unit_value)
            result.append(int2words(value) + " " + name)
        unit_value //= 1000
    if integer > 0:
        words = two_digit(integer) if integer < 100 else three_digit(integer)
        result.append(words)
    return " ".join(result)


def str2int(string: str) -> int:
    """
    introduction
    ==========
    Converts strings to integers and supports parsing ordinary floating-point numbers
    and scientific notation floating-point numbers.

    example
    ==========
    >>> str2int("0.123456789e+5")
    12345
    >>>
    :param string: string
    :return:
    """
    for item in string:
        if item not in "0123456789e.+-":
            raise ContentError
    length = 0
    string = string.lower()
    if "e" in string:
        if string.count("e") != 1:
            raise ContentError
        parts = string.split("e")
        integer_part = parts[0]
        exponent_part = parts[1]
        if not exponent_part:
            raise ContentError
        if "." in integer_part:
            if integer_part.count(".") != 1:
                raise ContentError
            integer_part, length = integer_part.replace(".", ""), len(integer_part.split(".")[1])
        integer_value = int(integer_part)
        exponent_value = int(exponent_part)
        if exponent_value > 0:
            integer_value *= 10 ** exponent_value
        elif exponent_value < 0:
            div = divmod(integer_value, 10 ** (-exponent_value))
            integer_value = div[0] if not div[1] or div[0] >= 0 else div[0] + 1
        if integer_value and length:
            div = divmod(integer_value, 10 ** length)
            return div[0] if not div[1] or div[0] >= 0 else div[0] + 1
        else:
            return integer_value
    else:
        if "." in string:
            if string.count(".") != 1:
                raise ContentError
            integer_part = string.split(".")[0]
            return int(integer_part) if integer_part else 0
        else:
            return int(string)


def int2roman(integer: int, overline: bool = True) -> str:
    """
    introduction
    ==========
    Convert natural numbers to Roman numerals and support with or without an overline.

    example
    ==========
    >>> int2roman(12345)
    'X̄ĪĪCCCXLV'
    >>>
    :param integer: integer
    :param overline: bool
    :return:
    """
    if not isinstance(integer, int):
        if isinstance(integer, float):
            integer = round(integer)
        else:
            raise TypeError("The number to be converted can only be a natural number")
    if integer < 0:
        raise ValueError("The number to be converted can only be a natural number")

    def int2roman_helper(number):
        roman = ""
        i = 0
        while number > 0:
            tmp = number // ROMAN_VALUES[i]
            roman += ROMAN_SYMBOLS[i] * tmp
            number -= ROMAN_VALUES[i] * tmp
            i += 1
        return roman

    def roman_overline(num):
        return "̄".join(int2roman_helper(num)) + "̄"

    if overline and integer >= 4000:
        thousands = integer // 1000
        remainder = integer % 1000
    else:
        thousands = 0
        remainder = integer
    roman_num = ""
    if thousands > 0:
        roman_num += roman_overline(thousands)
    roman_num += int2roman_helper(remainder)
    return roman_num


def roman2int(roman_num: str) -> int:
    """
    introduction
    ==========
    Convert Roman numerals to natural numbers and support the presence or absence of an overline.

    example
    ==========
    >>> roman2int("X̄ĪĪCCCXLV")
    12345
    >>>
    :param roman_num: string
    :return:
    """

    def roman2int_helper(number):
        part = 0
        last = 10000
        i = 0
        while i < len(number):
            if number[i:i + 2] in ROMAN_SYMBOLS:
                value = ROMAN_VALUES[ROMAN_SYMBOLS.index(number[i:i + 2])]
                if value > last:
                    raise ContentError
                else:
                    last = value
                part += value
                i += 2
            elif number[i] in ROMAN_SYMBOLS:
                value = ROMAN_VALUES[ROMAN_SYMBOLS.index(number[i])]
                if value > last:
                    raise ContentError
                else:
                    last = value
                part += value
                i += 1
            else:
                raise ContentError
        return part

    if "̄" in roman_num:
        index = roman_num.rfind("̄")
        high = roman_num[:index + 1]
        over = high[1::2]
        if over.count("̄") != len(over):
            raise ContentError
        return roman2int_helper(high.replace("̄", "")) * 1000 + roman2int_helper(roman_num[index + 1:])
    else:
        return roman2int_helper(roman_num)


def float2fraction(number: float, mixed: bool = False, error: float = 1e-15) -> tuple:
    """
    introduction
    ==========
    Convert floating-point numbers to fractions and support mixed numbers and false fractions.

    example
    ==========
    >>> float2fraction(3.141592653589793, False, 1e-6)
    (355, 113)
    >>>
    :param number: float
    :param mixed: bool
    :param error: float
    :return:
    """
    if number < 0:
        number = -number
        flag = True
    else:
        flag = False
    whole = int(number)
    number -= whole
    numerator = 0
    denominator = 1
    while abs(number - float(numerator) / float(denominator)) > error:
        if number > float(numerator) / float(denominator):
            numerator += 1
        else:
            denominator += 1
    if flag:
        whole = -whole
        numerator = -numerator
    return (whole, numerator, denominator) if mixed else (whole * denominator + numerator, denominator)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
