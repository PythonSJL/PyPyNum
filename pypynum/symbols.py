OPERATORS = ["**", "*", "//", "/", "%", "+", "-"]
BASIC = "%()*+-./0123456789"
ENGLISH = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
GREEK = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω"
VALID = BASIC + ENGLISH + GREEK


def parse_expr(expr: str) -> list:
    expr = expr.replace(" ", "")
    if any([item not in VALID for item in expr]):
        raise ValueError("Characters other than Arabic numerals, English letters, Greek letters, operators, "
                         "decimal points, and parentheses cannot appear in expressions")
    depth = 0
    pointer = 0
    result = []
    for index in range(len(expr)):
        if expr[index] == "(":
            if depth == 0:
                result.append(index)
            depth += 1
        elif expr[index] == ")":
            if depth == 0:
                raise ValueError("The parentheses in the expression are not paired")
            depth -= 1
            if depth == 0:
                sub = parse_expr(expr[result[-1] + 1:index])
                if not sub:
                    raise ValueError("The inside of the parentheses in an expression cannot be empty")
                result[-1] = sub
        elif depth == 0 and expr[index] in OPERATORS:
            number = expr[pointer:index]
            if number and "(" not in number:
                if number.count(".") > 1:
                    raise ValueError("Syntax error in expression")
                nan = True
                for item in number:
                    if nan and item.isdigit():
                        nan = False
                    elif not nan and not (item == "." or item.isdigit()):
                        raise NameError("The name of the algebra is invalid")
                result.append(number)
            if index != 0:
                pointer = index + 1
            else:
                pointer = index
            if expr[index] in "*/" and expr[index] == expr[index - 1]:
                if result[-1] in OPERATORS:
                    raise ValueError("Syntax error in expression")
                result.append(expr[index] * 2)
            elif index != 0 and expr[index] != expr[index + 1]:
                if result[-1] in OPERATORS:
                    raise ValueError("Syntax error in expression")
                result.append(expr[index])
    number = expr[pointer:]
    if number and "(" not in number:
        result.append(number)
    if depth != 0:
        raise ValueError("The parentheses in the expression are not paired")
    return result if len(result) != 1 else result[0]


# TODO 表达式展开
# TODO 表达式化简
# TODO 符号微分
# TODO 符号积分
...
