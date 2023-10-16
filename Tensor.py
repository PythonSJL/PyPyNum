class Tensor:
    def __init__(self, data):
        is_valid_tensor(data)
        self.shape = get_shape(data)
        self.data = data

    def add(self, tensor):
        if not isinstance(self.data, list) or not isinstance(tensor.data, list):
            return self.data + tensor.data
        if len(self.data) != len(tensor.data):
            raise ValueError("Tensor dimensions do not match")
        return Tensor([Tensor(t1).add(Tensor(t2)) for t1, t2 in zip(self.data, tensor.data)])

    def __add__(self, tensor):
        if isinstance(tensor, Tensor):
            return Tensor(tolist(self.add(tensor)))
        elif isinstance(tensor, int) or isinstance(tensor, float):
            return Tensor(tensor_and_number(self, "+", tensor))
        else:
            raise ValueError("The other must be a tensor or a number")

    def sub(self, tensor):
        if not isinstance(self.data, list) or not isinstance(tensor.data, list):
            return self.data - tensor.data
        if len(self.data) != len(tensor.data):
            raise ValueError("Tensor dimensions do not match")
        return Tensor([Tensor(t1).sub(Tensor(t2)) for t1, t2 in zip(self.data, tensor.data)])

    def __sub__(self, tensor):
        if isinstance(tensor, Tensor):
            return Tensor(tolist(self.sub(tensor)))
        elif isinstance(tensor, int) or isinstance(tensor, float):
            return Tensor(tensor_and_number(self, "-", tensor))
        else:
            raise ValueError("The other must be a tensor or a number")

    def mul(self, tensor):
        if not isinstance(self.data, list) or not isinstance(tensor.data, list):
            return self.data * tensor.data
        if len(self.data) != len(tensor.data):
            raise ValueError("Tensor dimensions do not match")
        return Tensor([Tensor(t1).mul(Tensor(t2)) for t1, t2 in zip(self.data, tensor.data)])

    def __mul__(self, tensor):
        if isinstance(tensor, Tensor):
            return Tensor(tolist(self.mul(tensor)))
        elif isinstance(tensor, int) or isinstance(tensor, float):
            return Tensor(tensor_and_number(self, "*", tensor))
        else:
            raise ValueError("The other must be a tensor or a number")

    def __getitem__(self, index):
        return self.data[index]

    def __str__(self):
        return str(self.data).replace("], ", "]\n").replace(",", "")

    def str(self):
        _max = len(max(str(self.data).replace("[", "").replace("]", "").replace(",", "").split(), key=len))
        _then = str(_format(self.data, _max)).replace("], ", "]\n").replace(",", "").replace("'", "").split("\n")
        _max = max([_.count("[") for _ in _then])
        return "\n".join([(_max - _.count("[")) * " " + _ + "\n" * (_.count("]") - 1) for _ in _then]).strip()


def is_valid_tensor(_tensor):
    if isinstance(_tensor, list):
        _length = len(_tensor)
        if _length:
            if all(isinstance(_, list) for _ in _tensor):
                _first = len(_tensor[0])
                for _ in _tensor:
                    if len(_) == _first:
                        is_valid_tensor(_)
                    else:
                        raise ValueError("The shape of the tensor is incorrect")
            elif not all(isinstance(_, (int, float, complex, Tensor)) for _ in _tensor):
                raise ValueError("The value of the tensor must be a number")
        else:
            raise ValueError("Tensor cannot be empty")
    elif not isinstance(_tensor, (int, float, complex)):
        raise ValueError("Tensor must use a list")


def get_shape(_tensor):
    _shape = []
    _sub = _tensor
    while isinstance(_sub, list):
        _shape.append(len(_sub))
        _sub = _sub[0]
    return _shape


def tolist(_nested_list):
    if isinstance(_nested_list, Tensor):
        _copy = []
        for item in _nested_list:
            _copy.append(tolist(item))
        return _copy
    else:
        return _nested_list


def _format(_nested_list, _max_length):
    if isinstance(_nested_list, list):
        _copy = []
        for item in _nested_list:
            _copy.append(_format(item, _max_length))
        return _copy
    else:
        _item = str(_nested_list)
        return " " * (_max_length - len(_item)) + _item


def tensor_and_number(tensor, operator, number):
    if isinstance(tensor, Tensor) or isinstance(tensor, list):
        _result = []
        for item in tensor:
            _result.append(tensor_and_number(item, operator, number))
        return _result
    else:
        if operator in ["+", "-", "*"]:
            return eval(f"{tensor} {operator} {number}")


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
    print(result.str())
    print("\nEND\n")
    result = tensor1 - tensor2
    print(result)
    print("\nEND\n")
    print(result.str())
    print("\nEND\n")
    result = tensor1 * tensor2
    print(result)
    print("\nEND\n")
    print(result.str())
    print("\nEND\n")
