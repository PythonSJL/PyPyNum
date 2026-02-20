import operator


class Group:
    def __init__(self, data, operation=operator.mul):
        self.__data = set(data)
        if not hasattr(operation, "__call__"):
            raise TypeError("The operation function must be callable")
        if hasattr(operation, "__code__") and operation.__code__.co_argcount != 2:
            raise TypeError("The operation function must have two arguments")
        self.__operation = operation
        self.__identity = None
        op = self.__operation
        for e in self.__data:
            try:
                is_identity = all([op(e, a) == a and op(a, e) == a for a in self.__data])
                if is_identity:
                    self.__identity = e
                    break
            except Exception:
                continue

    def __repr__(self):
        return "G{" + ", ".join(sorted(map(repr, self.__data))) + "}"

    def elements(self):
        return set(self.__data)

    def getop(self):
        return self.__operation

    def is_closed(self):
        for a in self.__data:
            for b in self.__data:
                try:
                    result = self.__operation(a, b)
                except Exception:
                    return False
                if result not in self.__data:
                    return False
        return True

    def is_associative(self):
        op = self.__operation
        for a in self.__data:
            for b in self.__data:
                for c in self.__data:
                    try:
                        res1 = op(op(a, b), c)
                        res2 = op(a, op(b, c))
                        if res1 != res2:
                            return False
                    except Exception:
                        return False
        return True

    @property
    def identity(self):
        return self.__identity

    def has_identity(self):
        return self.__identity is not None

    def has_inverses(self):
        identity = self.identity
        if identity is None:
            return False
        op = self.__operation
        for a in self.__data:
            found_inverse = False
            for b in self.__data:
                try:
                    if op(a, b) == identity and op(b, a) == identity:
                        found_inverse = True
                        break
                except Exception:
                    continue
            if not found_inverse:
                return False
        return True

    def is_semigroup(self):
        return self.is_closed() and self.is_associative()

    def is_monoid(self):
        return self.is_closed() and self.is_associative() and self.has_identity()

    def is_group(self):
        return self.is_closed() and self.is_associative() and self.has_identity() and self.has_inverses()

    is_magma = is_closed

    def is_quasigroup(self):
        if not self.is_closed():
            return False
        op = self.__operation
        elements = self.__data
        n = len(elements)
        for a in elements:
            row_products = set()
            col_products = set()
            for x in elements:
                try:
                    res_right = op(a, x)
                    res_left = op(x, a)
                except Exception:
                    return False
                if res_right in row_products or res_left in col_products:
                    return False
                row_products.add(res_right)
                col_products.add(res_left)
            if len(row_products) != n or len(col_products) != n:
                return False
        return True

    def is_loop(self):
        return self.has_identity() and self.is_quasigroup()

    def order(self):
        return len(self.__data)

    def is_supergroup(self, other):
        return other.is_subgroup(self)

    def is_subgroup(self, other):
        if not isinstance(other, Group):
            return False
        if not self.elements().issubset(other.elements()):
            return False
        op_other = other.getop()
        e_other = other.identity
        if e_other is None:
            return False
        if e_other not in self.__data:
            return False
        for a in self.__data:
            for b in self.__data:
                try:
                    res = op_other(a, b)
                except Exception:
                    return False
                if res not in self.__data:
                    return False
        for a in self.__data:
            has_inverse = False
            for b in self.__data:
                if op_other(a, b) == e_other and op_other(b, a) == e_other:
                    has_inverse = True
                    break
            if not has_inverse:
                return False
        return True

    def __eq__(self, other):
        if not isinstance(other, Group):
            return False
        if self.__data != other.__data:
            return False
        if self.__operation is other.__operation:
            return True
        op1_name = getattr(self.__operation, "__name__", None)
        op2_name = getattr(other.__operation, "__name__", None)
        return op1_name == op2_name

    def __ne__(self, other):
        return not self.__eq__(other)


def group(data, operation=operator.mul):
    return Group(data, operation)
