from .ufuncs import *


class Group:
    def __init__(self, data):
        self.__data = set(data)
        if any([any([not hasattr(elem, attr) for attr in ["__add__", "__sub__", "__mul__", "__truediv__"]])
                for elem in self.__data]):
            raise TypeError("All elements in the group must be computable")

    def __repr__(self):
        return "G{" + ", ".join(sorted([repr(elem) for elem in self.__data])) + "}"

    def is_closed(self, operation=multiply, modulus=None):
        for a in self.__data:
            for b in self.__data:
                if modulus is None:
                    if operation(a, b) not in self.__data:
                        return False
                elif isinstance(modulus, (int, float, complex)):
                    if operation(a, b) % modulus not in self.__data:
                        return False
        return True

    def is_associative(self, operation=multiply, modulus=None):
        for a in self.__data:
            for b in self.__data:
                for c in self.__data:
                    if modulus is None:
                        if operation(operation(a, b), c) != operation(a, operation(b, c)):
                            return False
                    elif isinstance(modulus, (int, float, complex)):
                        if operation(operation(a, b) % modulus, c) % modulus != operation(
                                a, operation(b, c) % modulus) % modulus:
                            return False
        return True

    def has_identity(self, operation=multiply):
        for a in self.__data:
            if any([operation(a, e) == a and operation(e, a) == a for e in self.__data]):
                return True
        return False

    def has_inverses(self, operation=multiply):
        for a in self.__data:
            for b in self.__data:
                if not any([operation(a, b) == e and operation(b, a) == e for e in self.__data]):
                    return False
        return True

    def is_semigroup(self, operation=multiply, modulus=None):
        return True if self.is_closed(operation, modulus) and self.is_associative(operation, modulus) else False

    def is_group(self, operation=multiply, modulus=None):
        return True if all([self.is_closed(operation, modulus), self.is_associative(operation, modulus),
                            self.has_identity(operation), self.has_inverses(operation)]) else False

    def order(self):
        return len(self.__data)

    def data(self):
        return self.__data.copy()

    def __eq__(self, other):
        return self.__data == other.__data

    def __ne__(self, other):
        return not self.__data == other.__data


def group(data):
    return Group(data)
