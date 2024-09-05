from .tools import linspace
from .types import Union, arr, real

thing = Union[list, str]


def color(text: str, rgb: arr) -> str:
    """
    To render colors to text.
    :param text: string
    :param rgb: list | tuple
    :return:
    """
    if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
        raise ValueError("RGB must be a triplet")
    red, green, blue = rgb
    if not (0 <= red <= 255 and 0 <= green <= 255 and 0 <= blue <= 255):
        raise ValueError("The valid range for RGB values is from 0 to 255")
    number = 16 + round(red / 51.2) * 36 + round(green / 51.2) * 6 + round(blue / 51.2)
    return "\x1b[38;5;{}m{}\x1b[m".format(number, text)


def change(data: thing) -> thing:
    """
    Transform the background between list and string forms.
    :param data: list | string
    :return:
    """
    if isinstance(data, list):
        return "\n".join(["".join(_) for _ in data])
    elif isinstance(data, str):
        return [list(_) for _ in data.split("\n")]
    raise TypeError("The input parameter type can only be a list or string")


def background(right: real = 5, left: real = -5, top: real = 5, bottom: real = -5,
               complexity: real = 5, ratio: real = 3, string: bool = False) -> thing:
    """
    Generate an empty coordinate system.
    :param right: integer | float
    :param left: integer | float
    :param top: integer | float
    :param bottom: integer | float
    :param complexity: integer | float
    :param ratio: integer | float
    :param string: bool.
    :return:
    """
    if abs(ratio) != ratio:
        raise ValueError("The ratio cannot be less than zero")
    if right - left < 1 / complexity or top - bottom < 1 / complexity:
        raise ValueError("The defined width or height cannot be less than the reciprocal of complexity")
    x = linspace(left, right, round((right - left) * complexity + 1))
    plane = [[" "] * 10 + ["|"] + [" "] * len(x) if _ != round((top - bottom) * complexity / 2 / ratio)
             else [" "] * 10 + ["|"] + list(" ".join("_" * (len(x) // 2 + 1))) for _ in range(
        round((top - bottom) * complexity / ratio))] + [[" "] * 10 + ["|"] + ["_"] * len(x)]
    plane[0][:10] = "{:.2e}".format(top).rjust(10)
    plane[-1][:10] = "{:.2e}".format(bottom).rjust(10)
    plane[round((top - bottom) * complexity / 2 / ratio)][:10] = "{:.2e}".format((top + bottom) / 2).rjust(10)
    plane.append([" "] * 11 + list("{:.2e}".format(left).ljust(10)) + list(
        "{:.2e}".format((right + left) / 2).center(len(x) - 20)) + list("{:.2e}".format(right).rjust(10)))
    return plane if not string else change(plane)


def unary(function, right: real = 5, left: real = -5, top: real = 5, bottom: real = -5, complexity: real = 5,
          ratio: real = 3, string: bool = True, basic: list = None, character: str = ".", data: bool = False,
          coloration=False) -> thing:
    """
    Draw a graph of a unary function.
    :param function: function
    :param right: integer | float
    :param left: integer | float
    :param top: integer | float
    :param bottom: integer | float
    :param complexity: integer | float
    :param ratio: integer | float
    :param string: bool.
    :param basic: list
    :param character: string
    :param data: bool.
    :param coloration: bool.
    :return:
    """
    if not isinstance(character, str) or (len(character) != 1 and not coloration):
        raise ValueError("The parameter character must be one character")
    x = linspace(left, right, round((right - left) * complexity + 1))
    y = list(map(function, x))
    plane = [_[:] for _ in basic] if basic else background(right, left, top, bottom, complexity, ratio)
    for i, d in enumerate(y):
        d = round((top - d) * complexity / ratio)
        if 0 <= d <= len(y) - 1:
            plane[d][i + 11] = character
    if string:
        plane = change(plane)
    return [plane, list(zip(x, y))] if data else plane


def binary(function, right: real = 5, left: real = -5, top: real = 5, bottom: real = -5, complexity: real = 5,
           ratio: real = 3, error=0, compare="==", string: bool = True, basic: list = None,
           character: str = ".", data: bool = False, coloration=False) -> thing:
    """
    Draw a graph of binary equations.
    :param function: function
    :param right: integer | float
    :param left: integer | float
    :param top: integer | float
    :param bottom: integer | float
    :param complexity: integer | float
    :param ratio: integer | float
    :param error: integer | float
    :param compare: string
    :param string: bool.
    :param basic: list
    :param character: string
    :param data: bool.
    :param coloration: bool.
    :return:
    """
    if not isinstance(character, str) or (len(character) != 1 and not coloration):
        raise ValueError("The parameter character must be one character")
    x = linspace(left, right, round((right - left) * complexity + 1))
    y = linspace(top, bottom, round((top - bottom) * complexity / ratio + 1))
    plane = [_[:] for _ in basic] if basic else background(right, left, top, bottom, complexity, ratio)
    coordinates = []
    for p1, c1 in enumerate(y):
        for p0, c0 in enumerate(x):
            if compare == "==":
                flag = abs(function(c0, c1)) <= error
            elif compare == "<=":
                flag = function(c0, c1) <= error
            elif compare == ">=":
                flag = function(c0, c1) >= error
            else:
                raise ValueError("The parameter used for comparison can only be '==' or '<=' or '>='")
            if flag:
                if data:
                    coordinates.append((c0, c1))
                plane[p1][p0 + 11] = character
    if string:
        plane = change(plane)
    return [plane, coordinates] if data else plane


def c_unary(function, projection: str = "ri", right: real = 5, left: real = -5, top: real = 5, bottom: real = -5,
            complexity: real = 5, ratio: real = 3, string: bool = True, basic: list = None, character: str = ".",
            data: bool = False, coloration=False) -> thing:
    """
    Draw a graph of a complex function of one variable.
    :param function: function
    :param projection: string
    :param right: integer | float
    :param left: integer | float
    :param top: integer | float
    :param bottom: integer | float
    :param complexity: integer | float
    :param ratio: integer | float
    :param string: bool.
    :param basic: list
    :param character: string
    :param data: bool.
    :param coloration: bool.
    :return:
    """
    if not isinstance(character, str) or (len(character) != 1 and not coloration):
        raise ValueError("The parameter character must be one character")
    x = linspace(left, right, round((right - left) * complexity + 1))
    y = linspace(top, bottom, round((top - bottom) * complexity / ratio + 1))
    plane = [_[:] for _ in basic] if basic else background(right, left, top, bottom, complexity, ratio)
    coordinates = [((c0, c1), function(complex(c0, c1))) for p1, c1 in enumerate(y) for p0, c0 in enumerate(x)]
    for x, y in coordinates:
        if projection == "ri":
            _c0, _c1 = y.real, y.imag
        else:
            raise ValueError("Other modes are currently not supported")
        c0, c1 = round((_c0 - left) * complexity), round((top - _c1) * complexity / ratio)
        if 0 <= c0 <= len(plane[0]) - 12 and 0 <= c1 <= len(plane) - 2:
            plane[c1][c0 + 11] = character
    if string:
        plane = change(plane)
    return [plane, coordinates] if data else plane
