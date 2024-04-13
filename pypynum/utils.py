from .types import num

SetError = TypeError("The other object must be an OrderedSet")


class OrderedSet:
    """
    It is an ordered set that supports various operations on ordinary sets, as well as other functions.
    :param sequence: A list or tuple
    """

    def __init__(self, sequence=None):
        self.__elements = []
        if sequence is not None:
            self.update(sequence)

    def add(self, element):
        if element not in self.__elements:
            self.__elements.append(element)

    def update(self, elements):
        if isinstance(elements, (list, tuple)):
            for element in elements:
                self.add(element)
        else:
            raise TypeError("Element sequences can only be one list or tuple")

    def insert(self, index, element):
        if element not in self.__elements:
            self.__elements.insert(index, element)

    def pop(self, index=-1):
        try:
            return self.__elements.pop(index)
        except IndexError:
            return None

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def to_list(self):
        from copy import deepcopy
        return list(deepcopy(self.__elements))

    def to_set(self):
        from copy import deepcopy
        return set(deepcopy(self.__elements))

    def hash(self):
        return list(map(hash, self.__elements))

    def find(self, element):
        try:
            return self.__elements.index(element)
        except ValueError:
            return -1

    def __mul__(self, other, replace=False):
        if isinstance(other, OrderedSet):
            if replace:
                self.__elements = [(item1, item2) for item1 in self for item2 in other]
            else:
                result = OrderedSet()
                result.__elements = [(item1, item2) for item1 in self for item2 in other]
                return result
        else:
            raise SetError

    cartesian_product = __mul__

    def __imul__(self, other):
        self.__mul__(other, True)
        return self

    def powerset(self):
        n = self.__len__()
        power_set = []
        for i in range(1 << n):
            subset = []
            for j in range(n):
                if i & 1 << j:
                    subset.append(self[j])
            result = OrderedSet()
            result.__elements = subset
            power_set.append(result)
        result = OrderedSet()
        result.__elements = power_set
        return result

    def partitions(self, k=None):
        def backtrack(start, path, res):
            if start == len(self):
                if not k or len(path) == k:
                    res.append(path)
                return
            backtrack(start + 1, path + [[self[start]]], res)
            for i, subset in enumerate(path):
                new_subset = subset + [self[start]]
                path_copy = path[:]
                path_copy[i] = new_subset
                backtrack(start + 1, path_copy, res)

        if not self.__elements:
            return [[]] if k is None or k == 0 else []
        result = []
        backtrack(0, [[]], result)
        final_result = []
        for p in result:
            if k is None or len(p) == k:
                temp_list = []
                for s in p:
                    temp = OrderedSet()
                    temp.__elements = s
                    temp_list.append(temp)
                final_result.append(temp_list)
        return sorted(final_result) if k is None else final_result

    def sort(self, key=None, reverse=False, new=True):
        try:
            if new:
                result = OrderedSet()
                result.__elements = sorted(self.__elements, key=key, reverse=reverse)
                return result
            self.__elements = sorted(self.__elements, key=key, reverse=reverse)
        except TypeError:
            return self

    def remove(self, element):
        self.__elements.remove(element)

    def clear(self):
        self.__elements = []

    def __contains__(self, element):
        return element in self.__elements

    def __getitem__(self, item):
        return self.__elements[item]

    def __len__(self):
        return len(self.__elements)

    def __sub__(self, other, replace=False):
        if isinstance(other, OrderedSet):
            result = list(filter(lambda e: e not in other.__elements, self.__elements))
            if replace:
                self.__elements = result
            else:
                return OrderedSet(result)
        else:
            raise SetError

    def __and__(self, other, replace=False):
        if isinstance(other, OrderedSet):
            total = self.__elements + other.__elements
            filtered = list(filter(lambda e: total.count(e) != 1, total))
            result = filtered[:len(filtered) // 2]
            if replace:
                self.__elements = result
            else:
                return OrderedSet(result)
        else:
            raise SetError

    def __or__(self, other, replace=False):
        if isinstance(other, OrderedSet):
            if replace:
                self.update(other.__elements)
            else:
                return OrderedSet(self.__elements + other.__elements)
        else:
            raise SetError

    def __matmul__(self, other, replace=False):
        if isinstance(other, OrderedSet):
            result = list(filter(lambda e: e not in self.__elements, other.__elements))
            if replace:
                self.__elements = result
            else:
                return OrderedSet(result)
        else:
            raise SetError

    def __xor__(self, other, replace=False):
        if isinstance(other, OrderedSet):
            total = self.__elements + other.__elements
            result = list(filter(lambda e: total.count(e) != 2, total))
            if replace:
                self.__elements = result
            else:
                return OrderedSet(result)
        else:
            raise SetError

    difference = __sub__
    intersection = __and__
    union = __or__
    complement = __matmul__
    symmetric_difference = __xor__

    def __isub__(self, other):
        self.__sub__(other, True)
        return self

    def __iand__(self, other):
        self.__and__(other, True)
        return self

    def __ior__(self, other):
        self.__or__(other, True)
        return self

    def __imatmul__(self, other):
        self.__matmul__(other, True)
        return self

    def __ixor__(self, other):
        self.__xor__(other, True)
        return self

    def __gt__(self, other):
        if not isinstance(other, OrderedSet):
            raise SetError
        return self.to_set() > other.to_set()

    def __ge__(self, other):
        if not isinstance(other, OrderedSet):
            raise SetError
        return self.to_set() >= other.to_set()

    def __eq__(self, other):
        if not isinstance(other, OrderedSet):
            raise SetError
        return self.to_set() == other.to_set()

    def __le__(self, other):
        if not isinstance(other, OrderedSet):
            raise SetError
        return self.to_set() <= other.to_set()

    def __lt__(self, other):
        if not isinstance(other, OrderedSet):
            raise SetError
        return self.to_set() < other.to_set()

    def __ne__(self, other):
        if not isinstance(other, OrderedSet):
            raise SetError
        return self.to_set() != other.to_set()

    def is_proper_superset(self, other):
        return self > other and self != other

    def is_superset(self, other):
        return self >= other

    def is_equal(self, other):
        return self == other

    def is_subset(self, other):
        return self <= other

    def is_proper_subset(self, other):
        return self < other and self != other

    def is_not_equal(self, other):
        return self != other

    def __hash__(self):
        return hash(self.__str__())

    def __repr__(self):
        return "OrderedSet({" + str(self.__elements)[1:-1] + "})"


class InfIterator:
    """
    Infinite sequence iterator, default to self increasing sequence.
    :param start: Initial value
    :param mode: Iterative mode
    :param common: Tolerance or common ratio
    """
    arithmetic = "arithmetic"
    geometric = "geometric"
    fibonacci = "fibonacci"
    catalan = "catalan"

    def __init__(self, start: num = 0, mode: str = "", common: num = 2):
        self.mode = mode
        self.common = common
        if self.mode == self.fibonacci:
            self.current = 0
        elif self.mode == self.catalan:
            self.current = 1
        else:
            self.current = start
        self.__next_value = 1 if self.mode == self.fibonacci or self.mode == self.catalan else 0

    def __iter__(self):
        return self

    def __next__(self):
        value = self.current
        if self.mode == self.arithmetic:
            self.current = self.current + self.common
        elif self.mode == self.geometric:
            self.current = self.current * self.common
        elif self.mode == self.fibonacci:
            self.__next_value, self.current = self.current + self.__next_value, self.__next_value
        elif self.mode == self.catalan:
            self.current = self.current * (4 * self.__next_value - 2) // (self.__next_value + 1)
            self.__next_value += 1
        else:
            self.current += 1
        return value

    def __repr__(self):
        return "InfIterator(" + self.mode + ")"
