num = int | float | complex


def product(numbers: list | tuple) -> num:
    """
    Product of multiple numbers
    >>> product([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    3628800
    >>>
    :param numbers: list | tuple
    :return:
    """
    result = 1
    for _ in numbers:
        result *= _
    return result


def sigma(i: int, n: int, f) -> num:
    """
    Σ
    >>> sigma(1, 10, lambda x: x ** x)
    10405071317
    >>>
    :param i: integer
    :param n: integer
    :param f: function
    :return:
    """
    return sum([f(_) for _ in range(i, n + 1)])


def pi(i: int, n: int, f) -> num:
    """
    Π
    >>> pi(1, 10, lambda x: x ** x)
    215779412229418562091680268288000000000000000
    >>>
    :param i: integer
    :param n: integer
    :param f: function
    :return:
    """
    return product([f(_) for _ in range(i, n + 1)])


def __parse(expression: str) -> list:
    """
    None
    :param expression: string
    :return:
    """
    length = len(expression)
    head = 0
    pointer = 0
    result = []
    while True:
        if pointer >= length - 1:
            if head <= length - 1:
                result.append(expression[head:])
            return result
        elif expression[pointer] == "(":
            tail = 0
            depth = 1
            for p in range(pointer + 1, length):
                if expression[p] == "(":
                    depth += 1
                elif expression[p] == ")":
                    depth -= 1
                if depth == 0:
                    tail = p
                    break
            result.append(__parse(expression[pointer + 1:tail]))
            head = tail + 1
            pointer = tail
        elif expression[pointer] in "+-":
            if head != pointer:
                result.append(expression[head:pointer])
                result.append(expression[pointer])
                head = pointer + 1
        elif expression[pointer] in "*/":
            if head != pointer:
                result.append(expression[head:pointer])
            if expression[pointer + 1] == expression[pointer]:
                result.append(expression[pointer:pointer + 2])
                pointer += 1
                head = pointer + 1
            else:
                result.append(expression[pointer:pointer + 1])
                head = pointer + 1
        pointer += 1


def parse(mathematical_expression: str) -> list:
    """
    Python Mathematical Expression Analysis
    >>> parse('(x + y - z) * (a - b + c)')
    [['x', '+', 'y', '-', 'z'], '*', ['a', '-', 'b', '+', 'c']]
    >>>
    :param mathematical_expression: string
    :return:
    """
    expression = mathematical_expression.replace(" ", "")
    depth = 0
    for p in range(len(expression)):
        if expression[p] == "(":
            depth += 1
        elif expression[p] == ")":
            depth -= 1
        if depth < 0:
            raise ValueError("The number of right parentheses is greater than the number of left parentheses")
    if depth > 0:
        raise ValueError("The number of left parentheses is greater than the number of right parentheses")
    return __parse(expression)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
