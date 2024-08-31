def pprint_matrix(matrix, style="numpy", output=True):
    if str(type(matrix)) != "<class 'pypynum.Matrix.Matrix'>":
        raise TypeError("The input must be of type 'pypynum.Matrix.Matrix'")
    style = style.strip().lower()
    supported_styles = ["numpy", "mpmath", "sympy", "borderless", "numbered"]
    if style not in supported_styles:
        raise ValueError("Unsupported style '{}'. Supported styles are: {}".format(style, supported_styles))
    max_length = max([len(str(item)) for row in matrix for item in row])
    if style == "numbered":
        max_length = max(max_length, len(str(matrix.cols)))
    format_str = "{:" + str(max_length) + "}"
    row_strings = []
    separator = "  " if style in ["mpmath", "sympy"] else " "
    for row in matrix:
        formatted_row = separator.join([format_str.format(item) for item in row])
        row_strings.append(formatted_row)
    result = ""
    if style == "numpy":
        result = str(matrix)
    elif style == "mpmath":
        result = "\n".join(["[" + row_str + "]" for row_str in row_strings])
    elif style == "sympy":
        border_length = max_length + 2
        column_count = matrix.cols
        middle_border = "⎢" + " " * (border_length * column_count - 2) + "⎥"
        end = len(row_strings) - 1
        if end < 1:
            result = "[" + row_strings[0] + "]"
        else:
            result = "⎡" + row_strings[0] + "⎤"
            for i, row_str in enumerate(row_strings[1:end]):
                result += "".join(["\n", middle_border, "\n", "⎢", row_str, "⎥"])
            result += "".join(["\n", middle_border, "\n", "⎣", row_strings[-1], "⎦"])
    elif style == "borderless":
        result = "\n".join(row_strings)
    elif style == "numbered":
        row_number_width, column_number_width = matrix.rows, matrix.cols
        row_number_format = "{:>" + str(len(str(row_number_width))) + "}"
        column_number_format = "{:^" + str(max_length) + "}"
        column_header = " " * len(str(row_number_width)) + " | " + " ".join(
            [column_number_format.format(str(j + 1)) for j in range(column_number_width)])
        for i, row in enumerate(row_strings):
            row_string = row_number_format.format((i + 1)) + " | " + row
            row_strings[i] = row_string
        result = "".join([column_header, "\n", "-" * len(row_strings[0]), "\n", "\n".join(row_strings)])
    if output:
        print(result)
    else:
        return result
