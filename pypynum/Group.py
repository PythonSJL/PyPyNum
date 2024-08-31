from .ufuncs import *


class Group:
    def __init__(self, data, operation=multiply):
        self.__data = set(data)
        self.__operation = None
        self.setop(operation)
        if any([any([not hasattr(elem, attr) for attr in ["__add__", "__sub__", "__mul__", "__truediv__"]])
                for elem in self.__data]):
            raise TypeError("All elements in the group must be computable")

    def __repr__(self):
        return "G{" + ", ".join(sorted([repr(elem) for elem in self.__data])) + "}"

    def elements(self):
        return set(self.__data)

    def getop(self):
        return self.__operation

    def setop(self, operation):
        if not hasattr(operation, "__call__"):
            raise TypeError("The operation function must be callable")
        if operation.__code__.co_argcount != 2:
            raise TypeError("The operation function must have two arguments")
        self.__operation = operation

    def is_closed(self, modulus=None):
        for a in self.__data:
            for b in self.__data:
                result = self.__operation(a, b)
                if modulus is not None:
                    result %= modulus
                if result not in self.__data:
                    return False
        return True

    def is_associative(self, modulus=None):
        for a in self.__data:
            for b in self.__data:
                for c in self.__data:
                    if modulus is None:
                        if self.__operation(self.__operation(a, b), c) != self.__operation(a, self.__operation(b, c)):
                            return False
                    else:
                        if self.__operation(self.__operation(a, b) % modulus, c) % modulus != self.__operation(
                                a, self.__operation(b, c) % modulus) % modulus:
                            return False
        return True

    def has_identity(self):
        identity = None
        for e in self.__data:
            if all([self.__operation(e, a) == a and self.__operation(a, e) == a for a in self.__data]):
                if identity is not None:
                    return False
                identity = e
        return identity is not None

    def has_inverses(self):
        identity = self.identity()
        if identity is None:
            return False
        for a in self.__data:
            for b in self.__data:
                if self.__operation(a, b) == identity and self.__operation(b, a) == identity:
                    break
            else:
                return False
        return True

    def identity(self):
        for e in self.__data:
            if all([self.__operation(e, a) == a and self.__operation(a, e) == a for a in self.__data]):
                return e
        return None

    def is_semigroup(self, modulus=None):
        return True if self.is_closed(modulus) and self.is_associative(modulus) else False

    def is_monoid(self, modulus=None):
        return True if self.is_closed(modulus) and self.is_associative(modulus) and self.has_identity() else False

    def is_group(self, modulus=None):
        return True if all([self.is_closed(modulus), self.is_associative(modulus),
                            self.has_identity(), self.has_inverses()]) else False

    def order(self):
        return len(self.__data)

    def is_supergroup(self, other, modulus=None):
        if not isinstance(other, Group):
            return False
        if not other.is_group(modulus):
            return False
        if self.getop() != other.getop():
            return False
        return self.elements().issuperset(other.elements())

    def is_subgroup(self, other, modulus=None):
        if not isinstance(other, Group):
            return False
        if not other.is_group(modulus):
            return False
        if self.getop() != other.getop():
            return False
        return self.elements().issubset(other.elements())

    def __eq__(self, other):
        return self.__data == other.__data

    def __ne__(self, other):
        return self.__data != other.__data


def group(data):
    return Group(data)
