from .Array import Array


class Tensor(Array):
    def add(self, tensor):
        if not isinstance(self.data, list) or not isinstance(tensor.data, list):
            return self.data + tensor.data
        if len(self.data) != len(tensor.data):
            raise ValueError("Tensor dimensions do not match")
        return Tensor([Tensor(t1, False).add(Tensor(t2, False)) for t1, t2 in zip(self.data, tensor.data)], False)

    def __add__(self, tensor):
        if isinstance(tensor, Tensor):
            return Tensor(tolist(self.add(tensor)), False)
        elif isinstance(tensor, (int, float, complex)):
            return Tensor(tensor_and_number(self, "+", tensor), False)
        else:
            raise ValueError("The other must be a tensor or a number")

    def sub(self, tensor):
        if not isinstance(self.data, list) or not isinstance(tensor.data, list):
            return self.data - tensor.data
        if len(self.data) != len(tensor.data):
            raise ValueError("Tensor dimensions do not match")
        return Tensor([Tensor(t1, False).sub(Tensor(t2, False)) for t1, t2 in zip(self.data, tensor.data)], False)

    def __sub__(self, tensor):
        if isinstance(tensor, Tensor):
            return Tensor(tolist(self.sub(tensor)), False)
        elif isinstance(tensor, (int, float, complex)):
            return Tensor(tensor_and_number(self, "-", tensor), False)
        else:
            raise ValueError("The other must be a tensor or a number")

    def mul(self, tensor):
        if not isinstance(self.data, list) or not isinstance(tensor.data, list):
            return self.data * tensor.data
        if len(self.data) != len(tensor.data):
            raise ValueError("Tensor dimensions do not match")
        return Tensor([Tensor(t1, False).mul(Tensor(t2, False)) for t1, t2 in zip(self.data, tensor.data)], False)

    def __mul__(self, tensor):
        if isinstance(tensor, Tensor):
            return Tensor(tolist(self.mul(tensor)), False)
        elif isinstance(tensor, (int, float, complex)):
            return Tensor(tensor_and_number(self, "*", tensor), False)
        else:
            raise ValueError("The other must be a tensor or a number")

    def matmul(self, tensor):
        from .Matrix import mat
        if not isinstance(self.data[0][0], list) or not isinstance(tensor.data[0][0], list):
            return (mat(self.data) @ mat(tensor.data)).data
        if len(self.data) != len(tensor.data):
            raise ValueError("Tensor dimensions do not match")
        return Tensor([Tensor(t1, False).matmul(Tensor(t2, False)) for t1, t2 in zip(self.data, tensor.data)], False)

    def __matmul__(self, tensor):
        if isinstance(tensor, Tensor):
            return Tensor(tolist(self.matmul(tensor)), False)
        else:
            raise ValueError("The other must be a tensor")


def tolist(_nested_list):
    if isinstance(_nested_list, Tensor):
        _copy = []
        for item in _nested_list:
            _copy.append(tolist(item))
        return _copy
    else:
        return _nested_list


def tensor_and_number(tensor, operator, number):
    if isinstance(tensor, Tensor) or isinstance(tensor, list):
        _result = []
        for item in tensor:
            _result.append(tensor_and_number(item, operator, number))
        return _result
    else:
        if operator in ["+", "-", "*"]:
            return eval("{} {} {}".format(tensor, operator, number))


def zeros(_dimensions):
    from .Array import zeros as _zeros
    return Tensor(_zeros(_dimensions))


def zeros_like(_nested_list):
    from .Array import zeros_like as _zeros_like
    return Tensor(_zeros_like(_nested_list.data))


def ten(data):
    return Tensor(data)


del Array
