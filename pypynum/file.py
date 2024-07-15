def read(file: str) -> list:
    """
    Read and parse files with the suffix ".pypynum"
    :param file: string
    :return:
    """
    exec("\n".join([
        "from .Array import Array",
        "from .Matrix import Matrix",
        "from .Tensor import Tensor",
        "from .Vector import Vector",
        "from .Geometry import Point, Line, Triangle, Quadrilateral, Polygon, Circle",
        "from .Group import Group",
        "from .Quaternion import Quaternion, Euler",
        "from .polynomial import Polynomial"
    ]))
    suffix = ".pypynum"
    if not file.endswith(suffix):
        raise FileExistsError("The file extension can only be '{}'".format(suffix))
    result = []
    with open(file, "r") as r:
        data = [item.split("\\") for item in r.read().split("\n") if item]
    for item in data:
        try:
            result.append(eval("{}(*{})".format(item[0], item[1])))
        except (IndexError, NameError, SyntaxError):
            raise FileExistsError("Data '{}' does not support processing".format("\\".join(item)))
    return result


def write(file: str, *cls: object):
    """
    Save data to a file with the suffix ".pypynum"
    :param file: string
    :param cls: instance
    :return:
    """
    from .Array import Array
    from .Geometry import Point, Line, Triangle, Quadrilateral, Polygon, Circle
    from .Group import Group
    from .Quaternion import Quaternion, Euler
    from .polynomial import Polynomial
    suffix = ".pypynum"
    if not file.endswith(suffix):
        raise FileExistsError("The file extension can only be '{}'".format(suffix))
    with open(file, "w") as w:
        for item in cls:
            _type = str(type(item))
            if "." not in _type:
                raise TypeError("Type '{}' does not support saving".format(type(item)))
            prefix = _type[_type.rfind(".") + 1:-2]
            if isinstance(item, Array):
                w.write("{}\\{}\n".format(prefix, str([item.data]).replace(" ", "")))
            elif isinstance(item, Point):
                w.write("{}\\{}\n".format(prefix, str([item.p]).replace(" ", "")))
            elif isinstance(item, Line):
                w.write("{}\\{}\n".format(prefix, str([item.a, item.b]).replace(" ", "")))
            elif isinstance(item, Triangle):
                w.write("{}\\{}\n".format(prefix, str([item.a, item.b, item.c]).replace(" ", "")))
            elif isinstance(item, Quadrilateral):
                w.write("{}\\{}\n".format(prefix, str([item.a, item.b, item.c, item.d]).replace(" ", "")))
            elif isinstance(item, Polygon):
                w.write("{}\\{}\n".format(prefix, str(item.points).replace(" ", "")))
            elif isinstance(item, Circle):
                w.write("{}\\{}\n".format(prefix, str([item.center, item.radius]).replace(" ", "")))
            elif isinstance(item, Group):
                w.write("{}\\{}\n".format(prefix, str([item.data()]).replace(" ", "")))
            elif isinstance(item, (Quaternion, Euler)):
                w.write("{}\\{}\n".format(prefix, str(item.data()).replace(" ", "")))
            elif isinstance(item, Polynomial):
                w.write("{}\\{}\n".format(prefix, str([item.terms]).replace(" ", "")))
            else:
                raise TypeError("Type '{}' does not support saving".format(type(item)))
