from .errors import LogicError

FullError = LogicError("The input of this logic component is full")
InputError = LogicError("The output port number of the connection must be an integer")


class Basic:
    def __init__(self, label):
        if not isinstance(label, str):
            raise TypeError("The type of label can only be a string")
        self.type = str(type(self))
        self.__label = label
        self.data = None
        self.order0 = 0
        self.order1 = 0
        self.order2 = 0
        self.order3 = 0

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.__label)

    def out(self):
        raise NotImplementedError

    def set_order0(self, data):
        if not isinstance(data, int):
            raise InputError
        self.order0 = data

    def set_order1(self, data):
        if isinstance(self, Unary):
            raise NotImplementedError
        if not isinstance(data, int):
            raise InputError
        self.order1 = data

    def set_order2(self, data):
        if isinstance(self, Unary) or isinstance(self, Binary):
            raise NotImplementedError
        if not isinstance(data, int):
            raise InputError
        self.order2 = data

    def set_order3(self, data):
        if isinstance(self, Unary) or isinstance(self, Binary) or isinstance(self, Ternary):
            raise NotImplementedError
        if not isinstance(data, int):
            raise InputError
        self.order3 = data


class Unary(Basic):
    def __init__(self, label, pin0=None):
        super().__init__(label)
        if pin0 is not None and not isinstance(pin0, int):
            raise InputError
        self.pin0 = pin0

    def set_pin0(self, data):
        if not isinstance(data, int) and not isinstance(data, Basic):
            raise InputError
        self.pin0 = data

    def out(self):
        self.data = self.pin0.out()[
            self.order0] if isinstance(self.pin0, Basic) else 0 if self.pin0 is None else self.pin0


class Binary(Basic):
    def __init__(self, label, pin0=None, pin1=None):
        super().__init__(label)
        if any(map(lambda item: item is not None and not isinstance(item, int), [pin0, pin1])):
            raise InputError
        self.pin0 = pin0
        self.pin1 = pin1

    def set_pin0(self, data):
        if not isinstance(data, int) and not isinstance(data, Basic):
            raise InputError
        self.pin0 = data

    def set_pin1(self, data):
        if not isinstance(data, int) and not isinstance(data, Basic):
            raise InputError
        self.pin1 = data

    def out(self):
        self.data = [
            self.pin0.out()[self.order0] if isinstance(self.pin0, Basic) else 0 if self.pin0 is None else self.pin0,
            self.pin1.out()[self.order1] if isinstance(self.pin1, Basic) else 0 if self.pin1 is None else self.pin1
        ]


class Ternary(Basic):
    def __init__(self, label, pin0=None, pin1=None, pin2=None):
        super().__init__(label)
        if any(map(lambda item: item is not None and not isinstance(item, int), [pin0, pin1, pin2])):
            raise InputError
        self.pin0 = pin0
        self.pin1 = pin1
        self.pin2 = pin2

    def set_pin0(self, data):
        if not isinstance(data, int) and not isinstance(data, Basic):
            raise InputError
        self.pin0 = data

    def set_pin1(self, data):
        if not isinstance(data, int) and not isinstance(data, Basic):
            raise InputError
        self.pin1 = data

    def set_pin2(self, data):
        if not isinstance(data, int) and not isinstance(data, Basic):
            raise InputError
        self.pin2 = data

    def out(self):
        self.data = [
            self.pin0.out()[self.order0] if isinstance(self.pin0, Basic) else 0 if self.pin0 is None else self.pin0,
            self.pin1.out()[self.order1] if isinstance(self.pin1, Basic) else 0 if self.pin1 is None else self.pin1,
            self.pin2.out()[self.order2] if isinstance(self.pin2, Basic) else 0 if self.pin2 is None else self.pin2
        ]


class Quaternary(Basic):
    def __init__(self, label, pin0=None, pin1=None, pin2=None, pin3=None):
        super().__init__(label)
        if any(map(lambda item: item is not None and not isinstance(item, int), [pin0, pin1, pin2, pin3])):
            raise InputError
        self.pin0 = pin0
        self.pin1 = pin1
        self.pin2 = pin2
        self.pin3 = pin3

    def set_pin0(self, data):
        if not isinstance(data, int) and not isinstance(data, Basic):
            raise InputError
        self.pin0 = data

    def set_pin1(self, data):
        if not isinstance(data, int) and not isinstance(data, Basic):
            raise InputError
        self.pin1 = data

    def set_pin2(self, data):
        if not isinstance(data, int) and not isinstance(data, Basic):
            raise InputError
        self.pin2 = data

    def set_pin3(self, data):
        if not isinstance(data, int) and not isinstance(data, Basic):
            raise InputError
        self.pin3 = data

    def out(self):
        self.data = [
            self.pin0.out()[self.order0] if isinstance(self.pin0, Basic) else 0 if self.pin0 is None else self.pin0,
            self.pin1.out()[self.order1] if isinstance(self.pin1, Basic) else 0 if self.pin1 is None else self.pin1,
            self.pin2.out()[self.order2] if isinstance(self.pin2, Basic) else 0 if self.pin2 is None else self.pin2,
            self.pin3.out()[self.order3] if isinstance(self.pin3, Basic) else 0 if self.pin3 is None else self.pin3
        ]


def connector(previous, latter):
    if not isinstance(previous, Basic) or not isinstance(latter, Basic):
        raise TypeError("The connected logical components must inherit from the Basic class")
    previous = previous
    latter = latter
    if isinstance(latter, Unary):
        if latter.pin0 is None:
            latter.set_pin0(previous)
        else:
            raise FullError
    elif isinstance(latter, Binary):
        if latter.pin0 is None:
            latter.set_pin0(previous)
        elif latter.pin1 is None:
            latter.set_pin1(previous)
        else:
            raise FullError
    elif isinstance(latter, Ternary):
        if latter.pin0 is None:
            latter.set_pin0(previous)
        elif latter.pin1 is None:
            latter.set_pin1(previous)
        elif latter.pin2 is None:
            latter.set_pin2(previous)
        else:
            raise FullError
    else:
        raise NotImplementedError


class NOT(Unary):
    def out(self):
        super().out()
        return [1 - self.data]


class AND(Binary):
    def out(self):
        super().out()
        return [1 if all(self.data) else 0]


class OR(Binary):
    def out(self):
        super().out()
        return [1 if any(self.data) else 0]


class NAND(Binary):
    def out(self):
        super().out()
        return [0 if all(self.data) else 1]


class NOR(Binary):
    def out(self):
        super().out()
        return [0 if any(self.data) else 1]


class XNOR(Binary):
    def out(self):
        super().out()
        return [1 if self.data[0] == self.data[1] else 0]


class XOR(Binary):
    def out(self):
        super().out()
        return [0 if self.data[0] == self.data[1] else 1]


class HalfAdder(Binary):
    def out(self):
        super().out()
        return [0 if self.data[0] == self.data[1] else 1, 1 if all(self.data) else 0]


class FullAdder(Ternary):
    def out(self):
        super().out()
        return [0 if sum(self.data) % 2 == 0 else 1, 1 if sum(self.data) > 1 else 0]


class HalfSuber(Binary):
    def out(self):
        super().out()
        return [0 if self.data[0] == self.data[1] else 1, 1 if not self.data[0] and self.data[1] else 0]


class FullSuber(Ternary):
    def out(self):
        super().out()
        return [0 if sum(self.data) % 2 == 0 else 1, 1 if self.data[0] - self.data[1] - self.data[2] < 0 else 0]


class TwoBMuler(Quaternary):
    def out(self):
        super().out()
        if self.data[0] == 0 and self.data[1] == 0 or self.data[2] == 0 and self.data[3] == 0:
            return [0, 0, 0, 0]
        elif self.data[0] == 1 and self.data[1] == 0:
            return [self.data[2], self.data[3], 0, 0]
        elif self.data[2] == 1 and self.data[3] == 0:
            return [self.data[0], self.data[1], 0, 0]
        elif self.data[0] == 0 and self.data[1] == 1:
            return [0, self.data[2], self.data[3], 0]
        elif self.data[2] == 0 and self.data[3] == 1:
            return [0, self.data[0], self.data[1], 0]
        else:
            return [1, 0, 0, 1]


class TwoBDiver(Quaternary):
    def out(self):
        super().out()
        if self.data[2] == 0 and self.data[3] == 0:
            return [1, 1, 1, 1]
        elif self.data[0] == 0 and self.data[1] == 0:
            return [0, 0, 0, 0]
        elif self.data[2] == 1 and self.data[3] == 0:
            return [self.data[0], self.data[1], 0, 0]
        elif self.data[2] == 0 and self.data[3] == 1:
            return [self.data[1], 0, self.data[0], 0]
        elif self.data == [1, 1, 1, 1]:
            return [1, 0, 0, 0]
        else:
            return [0, 0, self.data[0], self.data[1]]


class JKFF(Binary):
    def __init__(self, label, pin0=None, pin1=None, state=0):
        super().__init__(label, pin0, pin1)
        self.__state = state

    def out(self):
        super().out()
        j, k = self.data
        if j == 0 and k == 0:
            self.__state = self.__state
        elif j == 0 and k == 1:
            self.__state = 0
        elif j == 1 and k == 0:
            self.__state = 1
        elif j == 1 and k == 1:
            self.__state = 1 - self.__state
        return [self.__state]


class DFF(Unary):
    def __init__(self, label, pin0=None, state=0):
        super().__init__(label, pin0)
        self.__state = state

    def out(self):
        super().out()
        d = self.data
        if d == 0:
            self.__state = 0
        elif d == 1:
            self.__state = 1
        return [self.__state]


class TFF(Unary):
    def __init__(self, label, pin0=None, state=0):
        super().__init__(label, pin0)
        self.__state = state

    def out(self):
        super().out()
        t = self.data
        if t == 0:
            self.__state = self.__state
        elif t == 1:
            self.__state = 1 - self.__state
        return [self.__state]


class COMP(Binary):
    def out(self):
        super().out()
        return [1, 0, 0] if self.data[0] and not self.data[1] else [
            0, 0, 1] if not self.data[0] and self.data[1] else [0, 1, 0]
