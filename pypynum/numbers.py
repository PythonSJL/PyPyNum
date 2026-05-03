ContentError = ValueError("The content of the string is invalid")
__ROMAN_VALUES = (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1)
__ROMAN_SYMBOLS = ("M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I")


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


def words2int(words: str) -> int:
    """
    introduction
    ==========
    Convert natural language words into integers.

    example
    ==========
    >>> words2int("sixteen million seven hundred and seventy-seven thousand two hundred and sixteen")
    16777216
    >>>
    :param words: string representation of a number in words
    :return: integer
    """
    if not isinstance(words, str):
        raise TypeError("The input must be a string")
    ones = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
            "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
    tens = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
            "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
    big_units = ["quadragintillion", "novemtrigintillion", "octrigintillion", "septrigintillion", "sextrigintillion",
                 "quintrigintillion", "quattuortrigintillion", "trestrigintillion", "duotrigintillion",
                 "untrigintillion", "trigintillion", "novemvigintillion", "octovigintillion", "septenvigintillion",
                 "sexvigintillion", "quinvigintillion", "quattuorvigintillion", "trevigintillion", "duovigintillion",
                 "unvigintillion", "vigintillion", "novemdecillion", "octodecillion", "septendecillion", "sexdecillion",
                 "quindecillion", "quattuordecillion", "tredecillion", "duodecillion", "undecillion", "decillion",
                 "nonillion", "octillion", "septillion", "sextillion", "quintillion", "quadrillion", "trillion",
                 "billion", "million", "thousand"]
    unit_multipliers = {name: 10 ** (3 * (len(big_units) - i)) for i, name in enumerate(big_units)}
    words = words.strip().lower()
    if words == "zero":
        return 0
    negative = False
    if words.startswith("negative "):
        negative = True
        words = words[9:].strip()
    words = words.replace(" and ", " ")
    tokens = []
    for word in words.split():
        if "-" in word:
            tokens.extend(word.split("-"))
        else:
            tokens.append(word)
    result = 0
    current = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in unit_multipliers:
            result += current * unit_multipliers[token]
            current = 0
            i += 1
        elif token == "hundred":
            if i > 0 and tokens[i - 1] in ones:
                current -= ones[tokens[i - 1]]
                current += ones[tokens[i - 1]] * 100
            i += 1
        elif token in tens:
            current += tens[token]
            i += 1
            if i < len(tokens) and tokens[i] in ones:
                current += ones[tokens[i]]
                i += 1
        elif token in ones:
            current += ones[token]
            i += 1
        else:
            raise ValueError("Unknown token: " + token)
    result += current
    return -result if negative else result


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
            tmp = number // __ROMAN_VALUES[i]
            roman += __ROMAN_SYMBOLS[i] * tmp
            number -= __ROMAN_VALUES[i] * tmp
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
            if number[i:i + 2] in __ROMAN_SYMBOLS:
                value = __ROMAN_VALUES[__ROMAN_SYMBOLS.index(number[i:i + 2])]
                if value > last:
                    raise ContentError
                else:
                    last = value
                part += value
                i += 2
            elif number[i] in __ROMAN_SYMBOLS:
                value = __ROMAN_VALUES[__ROMAN_SYMBOLS.index(number[i])]
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
        sign = -1
        number = -number
    else:
        sign = 1
    whole = int(number)
    f = number - whole
    if f <= error:
        if mixed:
            return sign * whole, 0, 1
        return sign * whole, 1
    h_2, k_2 = 0, 1
    h_1, k_1 = 1, 0
    val = f
    while True:
        a = int(val)
        numerator = a * h_1 + h_2
        denominator = a * k_1 + k_2
        if abs(numerator / denominator - f) <= error:
            break
        remainder = val - a
        if remainder <= error:
            break
        val = 1.0 / remainder
        h_2, k_2 = h_1, k_1
        h_1, k_1 = numerator, denominator
    if mixed:
        return sign * whole, sign * numerator, denominator
    else:
        return sign * (whole * denominator + numerator), denominator


def parse_float(s: str) -> tuple:
    """
    Introduction
    ==========
    Parse a floating-point number string and return the sign bit, exponent, and digits.

    Example
    ==========
    >>> parse_float("123.456")
    ('0', '-3', '123456')
    >>> parse_float("-123.456e+7")
    ('1', '4', '123456')
    >>>
    :param s: The string representation of the floating-point number to be parsed.
    :return: A tuple containing the sign bit (0 for positive, 1 for negative), the exponent, and the digits.
    """
    sign, integer, fraction, exponent = "", "", "", ""
    s = str(s).strip().lower()
    try:
        floating = float(s)
    except ValueError:
        raise ValueError("The input string cannot be converted to a float")
    if floating != floating:
        raise ValueError("The input string represents a NaN (Not a Number)")
    if s in ("inf", "-inf"):
        raise ValueError("The input string represents an infinity")
    if s[0] in ["+", "-"]:
        sign, s = s[0], s[1:]
    if "e" in s or "E" in s:
        exp_index = s.index("e") if "e" in s else s.index("E")
        exponent, s = s[exp_index + 1:], s[:exp_index]
    if "." in s:
        dot_index = s.index(".")
        integer, fraction = s[:dot_index], s[dot_index + 1:].rstrip("0")
    else:
        integer = s
    sign = "1" if sign == "-" else "0"
    new_exponent = int(exponent) if exponent else 0
    if fraction:
        new_exponent -= len(fraction)
    digits = integer + fraction or "0"
    new_digits = digits.rstrip("0")
    new_exponent += len(digits) - len(new_digits)
    new_digits = new_digits.lstrip("0")
    if not new_digits:
        new_digits = "0"
    return sign, str(new_exponent), new_digits


def round_sigfig(number: str, n: int, scientific: bool = False) -> str:
    """
    Introduction
    ==========
    Round a floating-point number string to a specified number of significant figures.

    Example
    ==========
    >>> round_sigfig("-123.456e+7", 4)
    '-1235000000'
    >>> round_sigfig("-123.456e+7", 2, scientific=True)
    '-1.2e+9'
    >>>
    :param number: The string representation of the floating-point number to be rounded.
    :param n: The number of significant figures to keep (must be a positive integer).
    :param scientific: Whether to use scientific notation for the output. Defaults to False.
    :return: The rounded number as a string.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("The number of significant figures \"n\" must be a positive integer.")
    try:
        sign, exponent_str, digits_str = parse_float(number)
    except ValueError:
        raise ValueError("The input \"number\" string is not a valid floating-point number string.")
    if digits_str == "0":
        zero_str = "-0" if sign == "1" else "0"
        return zero_str
    exponent = int(exponent_str)
    digits_length = len(digits_str)
    if digits_length > n:
        round_digit_char = digits_str[n]
        main_part_str = digits_str[:n]
        if int(round_digit_char) >= 5:
            main_part_list = list(main_part_str)
            carry = 1
            for i in range(len(main_part_list) - 1, -1, -1):
                digit = int(main_part_list[i]) + carry
                if digit == 10:
                    main_part_list[i] = "0"
                    carry = 1
                else:
                    main_part_list[i] = str(digit)
                    carry = 0
                    break
            main_part_str = "".join(main_part_list)
            if carry == 1:
                main_part_str = "1" + main_part_str
        new_digits_str = main_part_str
        final_exponent = exponent + digits_length - 1
        if len(new_digits_str) > n:
            final_exponent += 1
    else:
        new_digits_str = digits_str.ljust(n, "0")
        final_exponent = exponent + digits_length - 1
    result_parts = []
    if sign == "1":
        result_parts.append("-")
    if scientific:
        first_digit = new_digits_str[0]
        remaining_digits = new_digits_str[1:n] if n > 1 else ""
        if remaining_digits:
            result_parts.extend((first_digit, ".", remaining_digits))
        else:
            result_parts.append(first_digit)
        exp_sign = "+" if final_exponent >= 0 else "-"
        exp_abs = str(abs(final_exponent))
        result_parts.extend(("e", exp_sign, exp_abs))
    else:
        shift = final_exponent - n + 1
        digits_len = len(new_digits_str)
        point_pos = digits_len + shift
        if point_pos >= digits_len:
            normal_str = new_digits_str + "0" * (point_pos - digits_len)
        elif point_pos > 0:
            normal_str = new_digits_str[:point_pos] + "." + new_digits_str[point_pos:]
        else:
            normal_str = "0." + "0" * (-point_pos) + new_digits_str
        result_parts.append(normal_str)
    final_str = "".join(result_parts)
    return final_str
