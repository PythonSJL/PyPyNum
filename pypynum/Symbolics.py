operators = ["**", "*", "//", "/", "%", "+", "-"]
basic = "%()*+-./0123456789"
english = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
greek = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω"
valid = basic + english + greek


def interpreter(expr: str) -> list:
    expr = expr.replace(" ", "")
    if any([item not in valid for item in expr]):
        raise ValueError("Characters other than Arabic numerals, English letters, Greek letters, operators, "
                         "decimal points, and parentheses cannot appear in expressions")
    if expr[0] not in "0123456789-(" or expr[-1] not in "0123456789)":
        raise ValueError("Syntax error in expression")
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
                sub = interpreter(expr[result[-1] + 1:index])
                if not sub:
                    raise ValueError("The inside of the parentheses in an expression cannot be empty")
                result[-1] = sub
        elif depth == 0 and expr[index] in operators:
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
                if result[-1] in operators:
                    raise ValueError("Syntax error in expression")
                result.append(expr[index] * 2)
            elif index != 0 and expr[index] != expr[index + 1]:
                if result[-1] in operators:
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
