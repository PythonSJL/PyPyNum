"""
Special mathematical characters
"""

div = "÷"
mul = "×"
overline = "̄"
sgn = "±"
strikethrough = "̶"
subscript = "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"
superscript = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻᵋᵝᵟᵠᶿ"
underline = "_"
arrow = (
    ("↖", "↑", "↗"),
    ("←", "⇌", "→"),
    ("↙", "↓", "↘"),
    ("↔", "⇋", "↕"),
    ("⇐", "⇔", "⇒")
)
tab = (
    ("┌", "┬", "┐"),
    ("├", "┼", "┤"),
    ("└", "┴", "┘"),
    ("─", "╭", "╮"),
    ("│", "╰", "╯")
)
pi = "Ππ𝜫𝝅𝝥𝝿𝞟𝞹Пп∏ϖ∐ℼㄇ兀"
notsign = "¬"
degree = "°"
permille = "‰"
permyriad = "‱"
prime = "′"
dprime = "″"
arc = "⌒"
ln = "㏑"
log = "㏒"
others = "".join(map(chr, range(0x2200, 0x2300)))


def int2superscript(standard_str: str) -> str:
    superscript_map = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
                       "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}
    return "".join([superscript_map.get(digit, digit) for digit in standard_str])


def superscript2int(superscript_str: str) -> str:
    standard_map = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
                    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
    return "".join([standard_map.get(char, char) for char in superscript_str])


def int2subscript(standard_str: str) -> str:
    subscript_map = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
                     "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉"}
    return "".join([subscript_map.get(digit, digit) for digit in standard_str])


def subscript2int(subscript_str: str) -> str:
    standard_map = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
                    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9"}
    return "".join([standard_map.get(char, char) for char in subscript_str])
