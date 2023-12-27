from .Matrix import mat


class Tensor:
    def __init__(self, data, check=True):
        self.shape = get_shape(data)
        if check and not isinstance(data, (int, float, complex)):
            is_valid_tensor(data, self.shape)
        self.data = data

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

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        def _format(_nested_list, _max_length):
            if isinstance(_nested_list, list):
                _copy = []
                for item in _nested_list:
                    _copy.append(_format(item, _max_length))
                return _copy
            else:
                _item = str(_nested_list)
                return " " * (_max_length - len(_item)) + _item

        _max = len(max(str(self.data).replace("[", "").replace("]", "").replace(",", "").split(), key=len))
        _then = str(_format(self.data, _max)).replace("], ", "]\n").replace(",", "").replace("'", "").split("\n")
        _max = max([_.count("[") for _ in _then])
        return "\n".join([(_max - _.count("[")) * " " + _ + "\n" * (_.count("]") - 1) for _ in _then]).strip()


def get_shape(_tensor):
    _shape = []
    _sub = _tensor
    while isinstance(_sub, list):
        _shape.append(len(_sub))
        _sub = _sub[0]
    return _shape


def is_valid_tensor(_tensor, _shape):
    if len(_shape) != 1 and not isinstance(_tensor, list):
        raise ValueError("The shape of the tensor is incorrect")
    if len(_shape) == 1:
        if not isinstance(_tensor, list):
            raise ValueError("The shape of the tensor is incorrect")
        _ = [type(_) for _ in _tensor]
        if not all(_type in [int, float, complex, Tensor] for _type in _):
            raise ValueError("The value of the tensor must be a number")
    elif len(_tensor) == _shape[0]:
        for _ in _tensor:
            is_valid_tensor(_, _shape[1:])
    else:
        raise ValueError("The shape of the tensor is incorrect")


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


def ten(data):
    return Tensor(data)


if __name__ == "__main__":
    tensor1 = Tensor([[[[[[7, 6], [3, 1]], [[2, 7], [9, 1]]], [[[0, 8], [1, 4]], [[8, 9], [1, 1]]]],
                       [[[[9, 8], [7, 9]], [[3, 7], [7, 2]]], [[[5, 3], [8, 4]], [[7, 5], [7, 3]]]]],
                      [[[[[2, 8], [2, 6]], [[7, 4], [8, 0]]], [[[7, 5], [9, 8]], [[5, 8], [4, 8]]]],
                       [[[[5, 6], [5, 4]], [[3, 8], [1, 9]]], [[[8, 5], [7, 5]], [[1, 5], [9, 9]]]]]])
    tensor2 = Tensor([[[[[[2, 3], [3, 0]], [[5, 3], [6, 5]]], [[[9, 1], [2, 2]], [[5, 6], [0, 9]]]],
                       [[[[4, 7], [7, 5]], [[4, 6], [5, 6]]], [[[2, 5], [7, 0]], [[3, 6], [5, 1]]]]],
                      [[[[[5, 5], [4, 4]], [[4, 8], [5, 3]]], [[[3, 0], [5, 8]], [[7, 3], [4, 6]]]],
                       [[[[5, 9], [5, 5]], [[6, 9], [5, 6]]], [[[8, 9], [6, 4]], [[9, 3], [3, 2]]]]]])
    print(tensor1)
    print("\nEND\n")
    print(tensor2)
    print("\nEND\n")
    print(tensor1.shape)
    print("\nEND\n")
    print(tensor2.shape)
    print("\nEND\n")
    result = tensor1 + tensor2
    print(result)
    print("\nEND\n")
    result = tensor1 - tensor2
    print(result)
    print("\nEND\n")
    result = tensor1 * tensor2
    print(result)
    print("\nEND\n")
    result = tensor1 @ tensor2
    print(result)
    print("\nEND\n")
