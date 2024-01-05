from .errors import LogicError

FullError = LogicError("The input of this logic component is full")
InputError = LogicError("The value of pin can only be zero or one")


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

    def __repr__(self):
        if self.type.startswith("<class 'pypynum.Logic."):
            return self.type[22:-2] + "({})".format(self.__label)
        return self.type + "({})".format(self.__label)

    def out(self):
        raise NotImplementedError

    def set_order0(self, data):
        if data not in [0, 1]:
            raise InputError
        self.order0 = data

    def set_order1(self, data):
        if isinstance(self, Unary):
            raise NotImplementedError
        if data not in [0, 1]:
            raise InputError
        self.order1 = data

    def set_order2(self, data):
        if isinstance(self, Unary) or isinstance(self, Binary):
            raise NotImplementedError
        if data not in [0, 1]:
            raise InputError
        self.order2 = data


class Unary(Basic):
    def __init__(self, label, pin=None):
        super().__init__(label)
        if pin not in [0, 1, None]:
            raise InputError
        self.pin = pin

    def set_pin(self, data):
        if data not in [0, 1, None] and not isinstance(data, Basic):
            raise InputError
        self.pin = data

    def out(self):
        self.data = self.pin.out()[self.order0] if isinstance(self.pin, Basic) else 0 if self.pin is None else self.pin


class Binary(Basic):
    def __init__(self, label, pin0=None, pin1=None):
        super().__init__(label)
        if pin0 not in [0, 1, None] or pin1 not in [0, 1, None]:
            raise InputError
        self.pin0 = pin0
        self.pin1 = pin1

    def set_pin0(self, data):
        if data not in [0, 1, None] and not isinstance(data, Basic):
            raise InputError
        self.pin0 = data

    def set_pin1(self, data):
        if data not in [0, 1, None] and not isinstance(data, Basic):
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
        if pin0 not in [0, 1, None] or pin1 not in [0, 1, None] or pin2 not in [0, 1, None]:
            raise InputError
        self.pin0 = pin0
        self.pin1 = pin1
        self.pin2 = pin2

    def set_pin0(self, data):
        if data not in [0, 1, None] and not isinstance(data, Basic):
            raise InputError
        self.pin0 = data

    def set_pin1(self, data):
        if data not in [0, 1, None] and not isinstance(data, Basic):
            raise InputError
        self.pin1 = data

    def set_pin2(self, data):
        if data not in [0, 1, None] and not isinstance(data, Basic):
            raise InputError
        self.pin2 = data

    def out(self):
        self.data = [
            self.pin0.out()[self.order0] if isinstance(self.pin0, Basic) else 0 if self.pin0 is None else self.pin0,
            self.pin1.out()[self.order1] if isinstance(self.pin1, Basic) else 0 if self.pin1 is None else self.pin1,
            self.pin2.out()[self.order2] if isinstance(self.pin2, Basic) else 0 if self.pin2 is None else self.pin2
        ]


def connector(previous, latter):
    if not isinstance(previous, Basic) or not isinstance(latter, Basic):
        raise TypeError("The connected logical components must inherit from the Basic class")
    previous = previous
    latter = latter
    if isinstance(latter, Unary):
        if latter.pin is None:
            latter.set_pin(previous)
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
