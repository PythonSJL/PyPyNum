from .types import Any, Callable, Union, arr, ite, real


def frange(start: real, stop: real, step: float = 1.0) -> list:
    """
    Introduction
    ==========
    Generates a list of numbers starting from `start` up to and including `stop` using a specified `step`.

    Example
    ==========
    >>> frange(0, 10, 1.5)
    [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0]
    >>>
    :param start: The starting value of the range.
    :param stop: The end value of the range, inclusive.
    :param step: The increment between each number in the range.
    :return: A list of floats representing the range.
    """
    if isinstance(step, int):
        step = float(step)
    result = [_ * step for _ in range(int(start / step), int(stop / step + 2))]
    return result


def linspace(start: real, stop: real, number: int) -> list:
    """
    Introduction
    ==========
    Generates a list of evenly spaced values starting from `start` up to and including `stop` with a specified number
    of samples.

    Example
    ==========
    >>> linspace(2, 3, 4)
    [2.0, 2.25, 2.5, 2.75]
    >>>
    :param start: The starting value of the range.
    :param stop: The end value of the range, inclusive.
    :param number: The number of samples to generate.
    :return: A list of floats representing the evenly spaced range.
    """
    if number == 1:
        return [start]
    elif number == 2:
        return [start, stop]
    else:
        step = (stop - start) / (number - 1)
        return [start + i * step for i in range(number)]


def geomspace(start: real, stop: real, number: int) -> list:
    """
    Introduction
    ==========
    Generates a list of evenly spaced geometric progression values starting from `start` up to and including `stop`
    with a specified number of samples.

    Example
    ==========
    >>> geomspace(2, 3, 4)
    [2.0, 2.25, 2.625, 3.0]
    >>>
    :param start: The starting value of the geometric progression.
    :param stop: The end value of the geometric progression, inclusive.
    :param number: The number of samples to generate.
    :return: A list of floats representing the geometric progression.
    """
    if number == 1:
        return [start]
    elif number == 2:
        return [start, stop]
    else:
        step = (stop / start) ** (1 / (number - 1))
        return [start * step ** i for i in range(number)]


def dedup(iterable: ite) -> ite:
    """
    Introduction
    ==========
    Removes duplicate elements from an iterable and returns a new iterable without duplicates.

    Example
    ==========
    >>> dedup(["T", "h", "i", "s", " ", "i", "s", " ", "a", " ", "l", "i", "s", "t"])
    ['T', 'h', 'i', 's', ' ', 'a', 'l', 't']
    >>>
    :param iterable: The iterable from which duplicates will be removed.
    :return: A new iterable without duplicates.
    """
    result = []
    for item in iterable:
        if item not in result:
            result.append(item)
    if isinstance(iterable, list):
        return result
    elif isinstance(iterable, tuple):
        return tuple(result)
    elif isinstance(iterable, str):
        return "".join(result)
    else:
        raise TypeError("Iterable can only be a list, tuple or str")


def classify(array: arr) -> dict:
    """
    Introduction
    ==========
    Classifies elements of an array based on their type and returns a dictionary mapping types to lists of elements
    of that type.

    Example
    ==========
    >>> classify((1, 2.3, 4 + 5j, 6 - 7j, 8.9, 0))
    {<class 'int'>: [1, 0], <class 'float'>: [2.3, 8.9], <class 'complex'>: [(4+5j), (6-7j)]}
    >>>
    :param array: The array to be classified.
    :return: A dictionary mapping types to lists of elements of that type.
    """
    result = {}
    for item in array:
        _type = type(item)
        if _type in result:
            result[_type].append(item)
        else:
            result[_type] = [item]
    return result


def split(iterable: ite, key: arr, retain: bool = False) -> list:
    """
    Introduction
    ==========
    Splits an iterable into sub-iterables based on a list of keys. The sub-iterables contain elements from the
    original iterable that match the keys.

    Example
    ==========
    >>> split((1, 2, 2, 3, 3, 2, 2, 1, 2, 3, 4, 5, 6, 7, 8), [1, 3, 5], retain=True)
    [(), (1,), (2, 2), (3,), (), (3,), (2, 2), (1,), (2,), (3,), (4,), (5,), (6, 7, 8)]
    >>>
    :param iterable: The iterable to be split.
    :param key: The list of keys to split the iterable on.
    :param retain: A boolean indicating whether to retain the keys in the resulting sub-iterables.
    :return: A list of sub-iterables split based on the keys.
    """
    key = dedup(key)
    if not isinstance(retain, bool):
        raise TypeError("Only Boolean values can be used to determine whether to retain the key")
    if not isinstance(key, (list, tuple)):
        raise TypeError("The parameter 'key' can only be a list or tuple")
    result = []
    pointer = 0
    if isinstance(iterable, str):
        if key == [""]:
            raise ValueError("When iterable is a string, the key cannot be a list of empty strings")
        while True:
            indexes = {}
            for k in key:
                index = iterable.find(k, pointer)
                if index != -1 and index not in indexes:
                    indexes[index] = k
            if indexes:
                index = min(indexes)
                result.append(iterable[pointer:index])
                pointer = index
                if retain is True:
                    result.append(indexes[index])
                pointer += len(indexes[index])
            else:
                break
        result.append(iterable[pointer:])
    elif isinstance(iterable, (list, tuple)):
        for item in range(len(iterable)):
            if iterable[item] in key:
                result.append(iterable[pointer:item])
                if isinstance(iterable, (list, tuple)):
                    pointer = item + 1
                if retain is True:
                    result.append(iterable[item:item + 1])
        result.append(iterable[pointer:])
    else:
        raise TypeError("Iterable can only be a list, tuple or str")
    return result


def primality(n: int, iter_num: int = 10) -> bool:
    """
    Introduction
    ==========
    This function checks if a number is a prime using the Miller-Rabin primality test,
    which is a probabilistic test to determine if a number is a probable prime.

    Example
    ==========
    >>> primality(2 ** 4423 - 1)
    True
    >>>
    :param n: The number to be tested for primality.
    :param iter_num: The number of iterations for the test, which affects the accuracy.
        More iterations reduce the probability of falsely identifying a composite number as prime.
    :return: True if the number is a probable prime, False otherwise.
    """
    from random import randint
    if n == 2:
        return True
    elif n & 1 == 0 or n < 2:
        return False
    m, s = n - 1, 0
    while m & 1 == 0:
        m >>= 1
        s += 1
    for _ in range(iter_num):
        b = pow(randint(2, n - 1), m, n)
        if b == 1 or b == n - 1:
            continue
        for __ in range(s - 1):
            b = b * b % n
            if b == n - 1:
                break
        else:
            return False
    return True


def primes(limit: int) -> list:
    """
    Introduction
    ==========
    This function generates a list of prime numbers up to a specified limit.

    Example
    ==========
    >>> primes(100)
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    >>>
    :param limit: The upper limit to which prime numbers are to be found (inclusive).
    :return: A list of prime numbers up to the specified limit.
    """
    result = list(range(limit + 1))
    result[1] = 0
    p = 2
    s = 4
    while s <= limit:
        if result[p]:
            for i in range(s, limit + 1, p):
                result[i] = 0
        p += 1
        s = p * p
    return [x for x in result if x != 0]


def semiprimes(limit: int) -> list:
    """
    Introduction
    ==========
    This function generates a list of semiprime numbers up to a specified limit.

    A semiprime is a natural number that is the product of two prime numbers.

    Example
    ==========
    >>> semiprimes(64)
    [4, 6, 9, 10, 14, 15, 21, 22, 25, 26, 33, 34, 35, 38, 39, 46, 49, 51, 55, 57, 58, 62]
    >>>
    :param limit: The upper limit to which semiprime numbers are to be found (inclusive).
    :return: A list of semiprime numbers up to the specified limit.
    """
    ps = primes(limit // 2)
    length = len(ps)
    result = []
    for i in range(length):
        first = ps[i]
        maximum = limit // first
        for j in range(i, length):
            second = ps[j]
            if second > maximum:
                break
            result.append(first * second)
    return sorted(result)


def twinprimes(limit: int) -> list:
    """
    Introduction
    ==========
    This function generates a list of twin prime pairs up to a specified limit.

    Twin primes are a pair of prime numbers that have a difference of two.

    Example
    ==========
    >>> twinprimes(100)
    [(3, 5), (5, 7), (11, 13), (17, 19), (29, 31), (41, 43), (59, 61), (71, 73)]
    >>>
    :param limit: The upper limit to which twin prime pairs are to be found (inclusive).
    :return: A list of twin prime pairs up to the specified limit.
    """
    ps = primes(limit)
    return [(p, q) for p, q in zip(ps, ps[1:]) if q - p == 2]


def prime_factors(integer: int, dictionary: bool = False, pollard_rho: bool = True) -> Union[list, dict]:
    """
    Introduction
    ==========
    This function computes the prime factors of a given integer.

    It can return the factors as a list or a dictionary, and it can use the Pollard's rho algorithm for factorization.

    Example
    ==========
    >>> prime_factors(2305567963945518424753102147331756070)
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    >>>
    :param integer: The integer to be factorized.
    :param dictionary: A boolean indicating whether to return the factors as a dictionary with the prime factors as
        keys and their counts as values.
    :param pollard_rho: A boolean indicating whether to use Pollard's rho algorithm for factorization. If False,
        a simple trial division method is used.
    :return: A list of prime factors or a dictionary with prime factors and their counts, depending on the
        'dictionary' parameter.
    """
    from math import gcd
    from random import randint

    def normal(n):
        factors = []
        divisor = 2
        while n > 1:
            tmp = divmod(n, divisor)
            if tmp[1]:
                divisor += 1
            else:
                factors.append(divisor)
                n = tmp[0]
        return factors

    def pollard(x, c):
        i, k = 1, 2
        x0 = randint(0, x)
        y = x0
        while 1:
            i += 1
            x0 = (x0 * x0 % x + c) % x
            d = gcd(y - x0, x)
            if d != 1 and d != x:
                return d
            if y == x0:
                return x
            if i == k:
                y = x0
                k += k

    def recursion(n):
        result = []
        if primality(n):
            return [n]
        p = n
        while p >= n:
            p = pollard(p, randint(1, n - 1))
        result.extend(recursion(p))
        result.extend(recursion(n // p))
        return result

    if integer > 0:
        output = recursion(integer) if pollard_rho else normal(integer)
        if dictionary:
            return {number: output.count(number) for number in set(output)}
        else:
            return sorted(output)
    else:
        return {} if dictionary else []


def magic_square(n: int) -> list:
    """
    Introduction
    ==========
    Generates a magic square of size n x n.

    A magic square is a square grid filled with distinct positive integers such that the sum of the numbers in each
    row, each column, and both main diagonals is the same number, known as the magic constant.

    Example
    ==========
    >>> magic_square(4)
    [[16, 2, 3, 13], [5, 11, 10, 8], [9, 7, 6, 12], [4, 14, 15, 1]]
    >>>
    :param n: The order of the magic square, which must be an integer greater than or equal to three.
    :return: A list of lists representing the generated magic square.
    """
    if not isinstance(n, int) or n < 3:
        raise ValueError("The order of the magic square must be an integer greater than or equal to three")
    result = [[0] * n for _ in range(n)]

    def odd():
        i, j = 0, n // 2
        for m in range(1, n * n + 1):
            result[i][j] = m
            new_i, new_j = (i - 1) % n, (j + 1) % n
            if result[new_i][new_j] != 0:
                new_i, new_j = (i + 1) % n, j
            i, j = new_i, new_j

    def single_even():
        half = n // 2
        left = magic_square(half)
        x = n ** 2 // 4
        k = (n - 2) // 4
        for i in range(half):
            for j in range(half):
                result[i][j] = left[i][j]
                result[i][j + half] = left[i][j] + 2 * x
                result[i + half][j] = left[i][j] + 3 * x
                result[i + half][j + half] = left[i][j] + x
        for i in range(half):
            for j in range(n):
                if j > (half + k + 1):
                    result[i][j], result[i + half][j] = result[i + half][j], result[i][j]
                if j < k:
                    result[i][j], result[i + half][j] = result[i + half][j], result[i][j]
        result[k][0], result[k + half][0] = result[k + half][0], result[k][0]
        result[k][k], result[k + half][k] = result[k + half][k], result[k][k]

    def double_even():
        c1 = 1
        c2 = n * n
        for i in range(n):
            for j in range(n):
                if i % 4 == j % 4 or (i + j) % 4 == 3:
                    result[i][j] = c2
                else:
                    result[i][j] = c1
                c2 -= 1
                c1 += 1

    if n & 1:
        odd()
    elif n & 3:
        single_even()
    else:
        double_even()
    return result


def levenshtein(x: ite, y: ite) -> int:
    """
    Introduction
    ==========
    Calculate the Levenshtein distance between two sequences.

    The Levenshtein distance is a measure of the difference between two sequences. It is defined as the minimum number
    of single-character edits (i.e., insertions, deletions, or substitutions) required to change one sequence into the
    other. This function supports any type of iterable sequences, such as strings, lists, or tuples.

    Example
    ==========
    >>> levenshtein("ensure", "nester")
    5
    >>>
    :param x: First sequence to compare.
    :param y: Second sequence to compare.
    :return: The Levenshtein distance between the two sequences.
    """
    if len(x) > len(y):
        x, y = y, x
    previous_row = tuple(range(len(x) + 1))
    for i2, c2 in enumerate(y):
        current_row = [i2 + 1]
        for i1, c1 in enumerate(x):
            insertions = previous_row[i1 + 1] + 1
            deletions = current_row[i1] + 1
            substitutions = previous_row[i1] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def lcsubseq(x: ite, y: ite) -> list:
    """
    Introduction
    ==========
    Calculate the Longest Common Subsequence (LCS) between two sequences.

    The Longest Common Subsequence is a sequence that appears in both given sequences in the same order, but not
    necessarily consecutively. This function uses dynamic programming to build a table that stores the lengths of the
    longest common subsequences of substring pairs, and then backtracks to construct the LCS.

    Example
    ==========
    >>> lcsubseq("ABCBDAB", "BDCAB")
    ['B', 'D', 'A', 'B']
    >>>
    :param x: First sequence to compare.
    :param y: Second sequence to compare.
    :return: The Longest Common Subsequence between the two sequences.
    """
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    i, j = m, n
    seq = []
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            seq.append(x[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    seq.reverse()
    return seq


def lcsubstr(x: ite, y: ite) -> list:
    """
    Introduction
    ==========
    Calculate the Longest Common Substring between two sequences.

    The Longest Common Substring is a substring that appears in both given sequences in the same order and is
    consecutive. This function uses dynamic programming to build a table that stores the lengths of the longest common
    substrings of substring pairs, and keeps track of the maximum length and its ending index to extract the substring.

    Example
    ==========
    >>> lcsubstr("ABCBDAB", "BDCAB")
    ['A', 'B']
    >>>
    :param x: First sequence to compare.
    :param y: Second sequence to compare.
    :return: The Longest Common Substring between the two sequences.
    """
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    end_idx = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_idx = i
            else:
                dp[i][j] = 0
    return list(x[end_idx - max_len:end_idx])


def cos_sim(seq1: ite, seq2: ite, is_vector: bool = False) -> float:
    """
    Introduction
    ==========
    Calculate the cosine similarity between two sequences.

    The cosine similarity is a measure of similarity between two non-zero vectors. It is defined as the cosine of the
    angle between them, which is computed as the dot product of the vectors divided by the product of their magnitudes.
    This function supports both numerical vectors and frequency distributions of sequences.

    Example
    ==========
    >>> cos_sim("hello world", "world hello")
    0.9999999999999998
    >>>
    :param seq1: First sequence to compare.
    :param seq2: Second sequence to compare.
    :param is_vector: A boolean indicating whether the input sequences are numerical vectors. Default is False.
    :return: The cosine similarity between the two sequences, ranging from -1 to 1.
    """
    from .maths import freq
    if is_vector:
        dot_product = sum([a * b for a, b in zip(seq1, seq2)])
        magnitude_seq1 = sum([a ** 2 for a in seq1]) ** 0.5
        magnitude_seq2 = sum([b ** 2 for b in seq2]) ** 0.5
    else:
        freq1 = freq(seq1)
        freq2 = freq(seq2)
        dot_product = sum([freq1.get(k, 0) * freq2.get(k, 0) for k in set(freq1) | set(freq2)])
        magnitude_seq1 = sum([v ** 2 for v in freq1.values()]) ** 0.5
        magnitude_seq2 = sum([v ** 2 for v in freq2.values()]) ** 0.5
    if magnitude_seq1 * magnitude_seq2 == 0:
        return 0.0
    return dot_product / (magnitude_seq1 * magnitude_seq2)


def kmp_table(pattern: ite) -> list:
    """
    Introduction
    ==========
    Generate the KMP (Knuth-Morris-Pratt) table for a given pattern.

    The KMP table is used to efficiently find occurrences of a pattern within a sequence by avoiding unnecessary
    comparisons after a mismatch. This table determines how many characters can be skipped after a mismatch.

    Example
    ========
    >>> kmp_table("AGCTGATCGTACGTAAGCTAGCTA")
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 2, 3, 4, 1, 2, 3, 4, 1]
    >>>
    :param pattern: The pattern for which to generate the KMP table.
    :return: A list representing the KMP table for the given pattern.
    """
    pattern_len = len(pattern)
    table = [0] * pattern_len
    pos, cnd = 1, 0
    while pos < pattern_len:
        if pattern[pos] == pattern[cnd]:
            cnd += 1
            table[pos] = cnd
            pos += 1
        elif cnd > 0:
            cnd = table[cnd - 1]
        else:
            table[pos] = 0
            pos += 1
    return table


def findall(seq: ite, pat: ite) -> list:
    """
    Introduction
    ==========
    Find all indices of the subsequence 'pat' in 'seq'.

    This function is designed to handle sequences such as lists, tuples, or strings and find all indices
    of specified subsequences. It allows overlapping matches.

    Example
    ========
    >>> findall([2, 1, 2, 1, 2, 1, 2, 1], [1, 2, 1, 2])
    [1, 3]
    >>>
    :param seq: The sequence in which to find the subsequence.
    :param pat: The subsequence to be found.
    :return: A list of starting indices where the subsequence is found.
    """
    if not isinstance(seq, (list, tuple, str)):
        raise TypeError("Parameter 'seq' must be a list, tuple, or string, got {}".format(type(seq)))
    if not isinstance(pat, (list, tuple, str)):
        raise TypeError("Parameter 'pat' must be a list, tuple, or string, got {}".format(type(pat)))
    if not pat:
        raise ValueError("Parameter 'pat' must not be empty")
    if isinstance(seq, str):
        seq = tuple(seq)
    if isinstance(pat, str):
        pat = tuple(pat)
    pat_len = len(pat)
    table = kmp_table(pat)
    indices = []
    i, j = 0, 0
    seq_len = len(seq)
    while i < seq_len:
        if j == pat_len:
            indices.append(i - j)
            j = table[j - 1]
        elif seq[i] == pat[j]:
            i += 1
            j += 1
        else:
            if j != 0:
                j = table[j - 1]
            else:
                i += 1
    if j == pat_len:
        indices.append(i - j)
    return indices


def replace(seq: arr, old: arr, new: arr, count: int = -1) -> arr:
    """
    Introduction
    ==========
    Replace occurrences of the subsequence 'old' in 'seq' with 'new'.

    This function is designed to handle sequences such as lists or tuples and replace specified subsequences
    with new ones. It also allows limiting the number of replacements.

    Example
    ==========
    >>> replace([1, 2, 3, 4, 2, 3], [2, 3], [5, 6])
    [1, 5, 6, 4, 5, 6]
    >>>
    :param seq: The sequence in which to replace the subsequence.
    :param old: The subsequence to be replaced.
    :param new: The subsequence to replace with.
    :param count: The maximum number of replacements to perform. Default is -1 (unlimited).
    :return: The modified sequence with replacements.
    """
    if not isinstance(seq, (list, tuple)):
        raise TypeError("Parameter 'seq' must be a list or a tuple, got {}".format(type(seq)))
    if not isinstance(old, (list, tuple)):
        raise TypeError("Parameter 'old' must be a list or a tuple, got {}".format(type(old)))
    if not isinstance(new, (list, tuple)):
        raise TypeError("Parameter 'new' must be a list or a tuple, got {}".format(type(new)))
    if not old:
        raise ValueError("Parameter 'old' must not be empty")
    if not isinstance(count, int) or count < 0 and count != -1:
        raise ValueError("Parameter 'count' must be a non-negative integer or -1, got {}".format(count))
    if count == 0:
        return seq[:]
    old_len = len(old)
    table = kmp_table(old)
    result = []
    i, j = 0, 0
    replaced = 0
    seq_len = len(seq)
    while i < seq_len:
        if j == old_len:
            result.extend(new)
            replaced += 1
            j = 0
            if count != -1 and replaced == count:
                result.extend(seq[i:])
                break
        elif seq[i] == old[j]:
            i += 1
            j += 1
        else:
            if j != 0:
                result.extend(old[:j])
                j = table[j - 1]
            else:
                result.append(seq[i])
                i += 1
    if j == old_len:
        result.extend(new)
    elif j > 0:
        result.extend(old[:j])
    return tuple(result) if isinstance(seq, tuple) else result


def fast_pow(a: Any, n: int, init: Any = 1, mul: Callable = None) -> Any:
    """
    Introduction
    ==========
    This function computes the power of a given base using the fast power algorithm.

    It is a versatile tool that can be used with any data type supporting multiplication, including the ability to
    perform matrix exponentiation by setting an initial matrix and a matrix multiplication function.
    For instance, to compute the matrix power of a 2x2 matrix, you can define the matrix as the base 'a',
    set an identity matrix as the 'init', and provide a function for matrix multiplication as 'mul'.

    Example
    ==========
    >>> fast_pow([[1, 2], [3, 4]], 31, init=[[1, 0], [0, 1]], mul=lambda _a, _b:
    ... [[_a[0][0] * _b[0][0] + _a[0][1] * _b[1][0], _a[0][0] * _b[0][1] + _a[0][1] * _b[1][1]],
    ... [_a[1][0] * _b[0][0] + _a[1][1] * _b[1][0], _a[1][0] * _b[0][1] + _a[1][1] * _b[1][1]]])
    [[10306408007805049875493, 15020838414172076413286], [22531257621258114619929, 32837665629063164495422]]
    >>>
    :param a: The base of the exponentiation, which can be any type of data.
    :param n: The exponent, which must be a non-negative integer.
    :param init: The initial value, which defaults to 1. It is used as the starting point for the computation.
    :param mul: A custom multiplication operation, which defaults to None. If provided, it should be a callable
                that takes two arguments and returns their product.
    :return: The result of the exponentiation, which will be of the same type as the 'init' parameter.
    """
    if mul is None:
        def mul(_a, _b):
            return _a * _b
    res = init
    while n:
        if n & 1:
            res = mul(res, a)
        a = mul(a, a)
        n >>= 1
    return res


def damerau(x: ite, y: ite) -> int:
    """
    Introduction
    ==========
    Calculate the Damerau-Levenshtein distance between two sequences.

    The Damerau-Levenshtein distance is a measure of the difference between two sequences. It is an extension of the
    Levenshtein distance that allows transpositions (i.e., swapping two adjacent characters) to be considered as a
    single edit operation. This function supports any type of iterable sequences, such as strings, lists, or tuples.

    Example
    ==========
    >>> damerau("ensure", "nester")
    3
    >>>
    :param x: First sequence to compare.
    :param y: Second sequence to compare.
    :return: The Damerau-Levenshtein distance between the two sequences.
    """
    if len(x) > len(y):
        x, y = y, x
    previous_row = tuple(range(len(x) + 1))
    two_rows_above = previous_row[:]
    for i2, c2 in enumerate(y):
        current_row = [i2 + 1]
        for i1, c1 in enumerate(x):
            insertions = previous_row[i1 + 1] + 1
            deletions = current_row[i1] + 1
            substitutions = previous_row[i1] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
            if i2 > 0 and i1 > 0 and x[i1] == y[i2 - 1] and x[i1 - 1] == y[i2] and x[i1] != y[i2]:
                current_row[-1] = min(current_row[-1], two_rows_above[i1 - 1] + 1)
        two_rows_above = previous_row
        previous_row = current_row
    return previous_row[-1]


def strip_helper(sequence: Any, keys_set: set, strip_start: bool, strip_end: bool) -> Any:
    """
    Introduction
    ==========
    Removes elements from the start and/or end of a sequence that match the specified keys.

    Example
    ==========
    >>> strip_helper([1, 2, 3, 4, 5, 1, 2], {1, 2}, True, True)
    [3, 4, 5]
    >>>
    :param sequence: The sequence from which to remove elements. It can be any iterable type.
    :param keys_set: A set of keys that, if present in the sequence, will be removed.
    :param strip_start: A boolean indicating whether to remove elements from the beginning of the sequence.
    :param strip_end: A boolean indicating whether to remove elements from the end of the sequence.
    :return: The sequence with the specified elements removed.
    """
    start, end = 0, len(sequence) - 1
    if strip_start:
        while start <= end and sequence[start] in keys_set:
            start += 1
    if strip_end:
        while end >= start and sequence[end] in keys_set:
            end -= 1
    return sequence[start:end + 1]


def strip(sequence: Any, keys: Any) -> Any:
    """
    Introduction
    ==========
    Removes elements from both the start and end of a sequence that match the specified keys.

    Example
    ==========
    >>> strip([1, 2, 3, 4, 5, 1, 2], [1, 2])
    [3, 4, 5]
    >>>
    :param sequence: The sequence from which to remove elements. It can be any iterable type.
    :param keys: A sequence of keys that, if present in the sequence, should be removed.
    :return: The sequence with the specified elements removed.
    """
    return strip_helper(sequence, set(keys), True, True)


def lstrip(sequence: Any, keys: Any) -> Any:
    """
    Introduction
    ==========
    Removes elements from the start of a sequence that match the specified keys.

    Example
    ==========
    >>> lstrip([1, 2, 3, 4, 5, 1, 2], [1, 2])
    [3, 4, 5, 1, 2]
    >>>
    :param sequence: The sequence from which to remove elements. It can be any iterable type.
    :param keys: A sequence of keys that, if present at the start of the sequence, should be removed.
    :return: The sequence with the specified elements removed from the start.
    """
    return strip_helper(sequence, set(keys), True, False)


def rstrip(sequence: Any, keys: Any) -> Any:
    """
    Introduction
    ==========
    Removes elements from the end of a sequence that match the specified keys.

    Example
    ==========
    >>> rstrip([1, 2, 3, 4, 5, 1, 2], [1, 2])
    [1, 2, 3, 4, 5]
    >>>
    :param sequence: The sequence from which to remove elements. It can be any iterable type.
    :param keys: A sequence of keys that, if present at the end of the sequence, should be removed.
    :return: The sequence with the specified elements removed from the end.
    """
    return strip_helper(sequence, set(keys), False, True)
