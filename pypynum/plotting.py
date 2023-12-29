from .tools import linspace

real = int | float
thing = list | str


def change(data: thing) -> thing:
    if isinstance(data, list):
        return "\n".join(["".join(_) for _ in data])
    elif isinstance(data, str):
        return [list(_) for _ in data.split("\n")]
    raise TypeError("The input parameter type can only be a list or string")


def background(right: real = 5, left: real = -5, top: real = 5, bottom: real = -5,
               complexity: real = 5, ratio: real = 3, merge: bool = False) -> thing:
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
    return plane if not merge else change(plane)


def unary(function, right: real = 5, left: real = -5, top: real = 5, bottom: real = -5, complexity: real = 5,
          ratio: real = 3, merge: bool = True, basic: list = None, character: str = ".", data: bool = False) -> thing:
    if not isinstance(character, str) or len(character) != 1:
        raise ValueError("The parameter character must be one character")
    x = linspace(left, right, round((right - left) * complexity + 1))
    y = list(map(function, x))
    plane = [_[:] for _ in basic] if basic else background(right, left, top, bottom, complexity, ratio)
    for i, d in enumerate(y):
        d = round((top - d) * complexity / ratio)
        if 0 <= d <= len(y) - 1:
            plane[d][i + 11] = character
    if merge:
        plane = change(plane)
    return [plane, list(zip(x, y))] if data else plane


def binary(function, right: real = 5, left: real = -5, top: real = 5, bottom: real = -5, complexity: real = 5,
           ratio: real = 3, error=0, compare="==", merge: bool = True, basic: list = None,
           character: str = ".", data: bool = False) -> thing:
    if not isinstance(character, str) or len(character) != 1:
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
    if merge:
        plane = change(plane)
    return [plane, coordinates] if data else plane


def c_unary(function, start: real, end: real, interval: real = 5, projection: str = "ri", right: real = 5,
            left: real = -5, top: real = 5, bottom: real = -5, complexity: real = 5, ratio: real = 3,
            merge: bool = True, basic: list = None, character: str = ".", data: bool = False) -> thing:
    if abs(interval) != interval:
        raise ValueError("The interval cannot be less than zero")
    if not isinstance(character, str) or len(character) != 1:
        raise ValueError("The parameter character must be one character")
    if projection == "ri":
        x = linspace(start, end, round((end - start) * interval + 1))
    else:
        x = linspace(left, right, round((right - left) * complexity + 1))
    c = [complex(_) for _ in map(function, x)]
    plane = [_[:] for _ in basic] if basic else background(right, left, top, bottom, complexity, ratio)
    coordinates = []
    for p, _ in enumerate(c):
        if projection == "ri":
            _c0, _c1 = _.real, _.imag
        elif projection == "xr":
            _c0, _c1 = p, _.real
        elif projection == "xi":
            _c0, _c1 = p, _.imag
        else:
            raise ValueError("The parameter used for section can only be 'ri' or 'xr' or 'xi'")
        c0, c1 = round((_c0 - left) * complexity) if projection == "ri" else _c0, round(
            (top - _c1) * complexity / ratio)
        if 0 <= c0 <= len(plane[0]) - 12 and 0 <= c1 <= len(plane) - 2:
            if data:
                coordinates.append((x[p], _.real, _.imag))
            plane[c1][c0 + 11] = character
    if merge:
        plane = change(plane)
    return [plane, coordinates] if data else plane
