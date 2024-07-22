from .confs import config
from .types import num

SetError = TypeError("The other object must be an OrderedSet")


class OrderedSet:
    def __init__(self, sequence=None):
        """
        It is an ordered set that supports various operations on ordinary sets, as well as other functions.
        :param sequence: A list or tuple
        """
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
        return self > other

    def is_superset(self, other):
        return self >= other

    def is_equal(self, other):
        return self == other

    def is_subset(self, other):
        return self <= other

    def is_proper_subset(self, other):
        return self < other

    def is_not_equal(self, other):
        return self != other

    def __hash__(self):
        return hash(self.__str__())

    def __repr__(self):
        return "OrderedSet({" + str(self.__elements)[1:-1] + "})"


class InfIterator:
    arithmetic = "arithmetic"
    geometric = "geometric"
    fibonacci = "fibonacci"
    catalan = "catalan"

    def __init__(self, start: num = 0, mode: str = "", common: num = 2):
        """
        Infinite sequence iterator, default to self increasing sequence.
        :param start: Initial value
        :param mode: Iterative mode
        :param common: Tolerance or common ratio
        """
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


class LinkedListNode:
    def __init__(self, value, next_node=None):
        """
        The nodes of a linked list can be connected by setting up successor nodes.
        :param value: The value of the node
        :param next_node: Next node object
        """
        self.value = value
        self.next_node = next_node


class LinkedList:
    def __init__(self):
        """
        It is a linked list that supports many functions.
        """
        self.head = None

    def length(self):
        result = 0
        current = self.head
        while current is not None:
            result += 1
            current = current.next_node
        return result

    def insert_beg(self, value):
        new_node = LinkedListNode(value)
        new_node.next_node = self.head
        self.head = new_node

    def insert_end(self, value):
        if self.head is None:
            self.head = LinkedListNode(value)
        else:
            current = self.head
            while current.next_node is not None:
                current = current.next_node
            current.next_node = LinkedListNode(value)

    def insert_idx(self, index, value):
        if index < 0:
            raise IndexError("Index must be non-negative")
        if index == 0:
            self.insert_beg(value)
            return
        current = self.head
        position = 0
        while current is not None and position < index - 1:
            current = current.next_node
            position += 1
        if not current:
            raise IndexError("Index out of range")
        new_node = LinkedListNode(value)
        new_node.next_node = current.next_node
        current.next_node = new_node

    def insert_after(self, target_value, new_value):
        if self.head is None:
            raise ValueError("The linked list is empty")
        current = self.head
        found = False
        while current is not None and not found:
            if current.value == target_value:
                found = True
            else:
                current = current.next_node
        if not found:
            raise ValueError("No node with the specified value was found")
        new_node = LinkedListNode(new_value)
        new_node.next_node = current.next_node
        current.next_node = new_node

    def replace(self, target_value, new_value, n=1):
        current = self.head
        count = 0
        while current is not None and count < n:
            if current.value == target_value:
                current.value = new_value
                count += 1
            current = current.next_node

    def delete_node(self, value):
        if self.head is None:
            return
        if self.head.value == value:
            self.head = self.head.next_node
            return
        current = self.head
        while current.next_node is not None:
            if current.next_node.value == value:
                current.next_node = current.next_node.next_node
                return
            current = current.next_node

    def contains(self, value):
        current = self.head
        while current is not None:
            if current.value == value:
                return True
            current = current.next_node
        return False

    def count(self, value):
        result = 0
        current = self.head
        while current is not None:
            if current.value == value:
                result += 1
            current = current.next_node
        return result

    def get_successor(self, value):
        current = self.head
        while current is not None and current.value != value:
            current = current.next_node
        if current is not None and current.next_node is not None:
            return current.next_node.value

    def to_list(self):
        result = []
        current = self.head
        while current:
            result.append(current.value)
            current = current.next_node
        return result

    def __len__(self):
        return self.length()

    def __contains__(self, item):
        return self.contains(item)

    def __repr__(self):
        current = self.head
        linked_list_str = ""
        while current is not None:
            linked_list_str += str(current.value)
            if current.next_node:
                linked_list_str += " -> "
            current = current.next_node
        return linked_list_str


class IntervalSet:
    def __init__(self, intervals=None):
        """
        It is a set of intervals and supports multiple operations.
        :param intervals: A sequence containing multiple intervals
        """
        self.intervals = []
        if intervals:
            self.add_intervals(intervals)

    def add_intervals(self, intervals):
        for interval in sorted(intervals, key=lambda x: x[0]):
            self.add_interval(*interval)

    def add_interval(self, start, end, start_open=False, end_open=False):
        if start == float("-inf") or start == float("inf"):
            start_open = True
        if end == float("-inf") or end == float("inf"):
            end_open = True
        start_open = bool(start_open)
        end_open = bool(end_open)
        if start > end or (start == end and (start_open or end_open)) or start == end == float(
                "inf") or start == end == float("-inf"):
            return
        intervals = []
        added = False
        for s, e, so, eo in self.intervals:
            if s <= start < e or s < end <= e or start <= s < end or start < e <= end or start == e and not (
                    start_open and eo) or s == end and not (so and end_open):
                start = min(s, start)
                end = max(e, end)
                start_open = start_open if start < s else so if start > s else start_open and so
                end_open = end_open if end > e else eo if end < e else end_open and eo
            elif start > e:
                intervals.append((s, e, so, eo))
            else:
                if not added:
                    intervals.append((start, end, start_open, end_open))
                    added = True
                intervals.append((s, e, so, eo))
        if not added:
            intervals.append((start, end, start_open, end_open))
        self.intervals = intervals

    def remove_interval(self, start, end, start_open=False, end_open=False):
        intervals = []
        start_open = bool(start_open)
        end_open = bool(end_open)
        for s, e, so, eo in self.intervals:
            if e <= start:
                intervals.append((s, e, so, eo))
            elif s >= end:
                intervals.append((s, e, so, eo))
            else:
                if s < start:
                    intervals.append((s, start, so, not start_open))
                if e > end:
                    intervals.append((end, e, not end_open, eo))
        self.intervals = intervals

    def is_superset(self, other):
        def contains_interval(a, b):
            s1, e1, so1, eo1 = a
            s2, e2, so2, eo2 = b
            return (s2 <= s1 if not so2 else s2 < s1) and (e1 <= e2 if not eo1 else e1 < e2)

        for interval in other.intervals:
            if not any([contains_interval(interval, other_interval) for other_interval in self.intervals]):
                return False
        return True

    def is_proper_superset(self, other):
        return self.is_superset(other) and self.is_not_equal(other)

    def is_equal(self, other):
        return self.intervals == other.intervals

    def is_proper_subset(self, other):
        return other.is_proper_superset(self)

    def is_subset(self, other):
        return other.is_superset(self)

    def is_not_equal(self, other):
        return self.intervals != other.intervals

    __gt__ = is_proper_superset
    __ge__ = is_superset
    __eq__ = is_equal
    __le__ = is_subset
    __lt__ = is_proper_subset
    __ne__ = is_not_equal

    def intersection(self, other):
        result = IntervalSet()
        i, j = 0, 0
        while i < len(self.intervals) and j < len(other.intervals):
            s1, e1, so1, eo1 = self.intervals[i]
            s2, e2, so2, eo2 = other.intervals[j]
            start = max(s1, s2)
            end = min(e1, e2)
            start_open = so1 if s1 > s2 else so2 if s1 < s2 else so1 or so2
            end_open = eo1 if e1 < e2 else eo2 if e1 > e2 else eo1 or eo2
            if start < end or (start == end and not (start_open and end_open)):
                result.add_interval(start, end, start_open, end_open)
            if e1 <= e2:
                i += 1
            if e2 <= e1:
                j += 1
        return result

    def union(self, other):
        result = IntervalSet(self.intervals)
        for interval in other.intervals:
            result.add_interval(*interval)
        return result

    def difference(self, other):
        result = IntervalSet(self.intervals)
        for s, e, so, eo in other.intervals:
            result.remove_interval(s, e, so, eo)
        return result

    def complement(self, other):
        return other.difference(self)

    def symmetric_difference(self, other):
        return self.union(other).difference(self.intersection(other))

    __and__ = intersection
    __or__ = union
    __sub__ = difference
    __matmul__ = complement
    __xor__ = symmetric_difference

    def latex(self):
        return self.__repr__(True)

    def __repr__(self, use_latex=False):
        use_latex = config.use_latex or use_latex
        intervals = self.intervals
        n = len(intervals)
        if n == 0:
            if use_latex:
                return "\\emptyset"
            elif config.use_unicode:
                return "∅"
            else:
                return "EmptySet"

        def format_interval(s, e, so, eo):
            infinity = "\\infty" if use_latex else "∞" if config.use_unicode else "inf"
            if s == e:
                if use_latex:
                    return "\\left\\{" + str(s) + "\\right\\}"
                else:
                    return "{{{}}}".format(s)
            else:
                start_str = str(s).replace("inf", infinity)
                end_str = str(e).replace("inf", infinity)
                if use_latex:
                    start_bracket = "\\left(" if so else "\\left["
                    end_bracket = "\\right)" if eo else "\\right]"
                else:
                    start_bracket = "(" if so else "["
                    end_bracket = ")" if eo else "]"
                return "{}{}, {}{}".format(start_bracket, start_str, end_str, end_bracket)

        formatted_intervals = [format_interval(*interval) for interval in intervals]
        union = " \\cup " if use_latex else " ∪ " if config.use_unicode else " U "
        return union.join(formatted_intervals)
