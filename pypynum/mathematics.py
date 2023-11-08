num = int | float | complex


def ptp(numbers: list | tuple) -> num:
    """
    Range of numbers
    >>> ptp([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    9
    >>>
    :param numbers: list | tuple
    :return:
    """
    return max(numbers) - min(numbers)


def median(numbers: list | tuple) -> num:
    """
    Median of numbers
    >>> median([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    5.5
    >>>
    :param numbers: list | tuple
    :return:
    """
    data = sorted(numbers)
    n = len(data)
    if n % 2 == 0:
        return (data[n // 2 - 1] + data[n // 2]) / 2
    else:
        return data[n // 2]


def freq(data: list | tuple):
    """
    Frequency of data
    >>> freq([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}
    >>>
    :param data: list | tuple
    :return:
    """
    result = {}
    for d in data:
        if d in result:
            result[d] += 1
        else:
            result[d] = 1
    return result


def mode(data: list | tuple):
    """
    Mode of data
    >>> mode([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    1
    >>>
    :param data: list | tuple
    :return:
    """
    count_dict = freq(data)
    max_count = 0
    max_num = 0
    for k, v in count_dict.items():
        if v > max_count:
            max_count = v
            max_num = k
    return max_num


def mean(numbers: list | tuple) -> num:
    """
    Average of numbers
    >>> mean([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    5.5
    >>>
    :param numbers: list | tuple
    :return:
    """
    return sum(numbers) / len(numbers)


def var(numbers: list | tuple) -> num:
    """
    Variance of numbers
    >>> var([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    8.25
    >>>
    :param numbers: list | tuple
    :return:
    """
    avg = mean(numbers)
    return sum([(x - avg) ** 2 for x in numbers]) / len(numbers)


def std(numbers: list | tuple) -> num:
    """
    Standard deviation of numbers
    >>> std([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    2.8722813232690143
    >>>
    :param numbers: list | tuple
    :return:
    """
    return var(numbers) ** 0.5


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
    >>> parse("(x + y - z) * (a - b + c)")
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
