from .errors import ContentError

ContentError = ContentError("The input string is invalid")
ORD_A = 65
ORD_a = 97
MORSE_CODE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.", "H": "....", "I": "..",
    "J": ".---", "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--", "Z": "--..", "0": "-----",
    "1": ".----", "2": "..---", "3": "...--", "4": "....-", "5": ".....", "6": "-....", "7": "--...", "8": "---..",
    "9": "----.", ".": ".-.-.-", ",": "--..--", "?": "..--..", "!": "-.-.--", "'": '.----.', "/": "-..-.", "(": "-.--.",
    ")": "-.--.-", "&": ".-...", ":": "---...", ";": "-.-.-.", "=": "-...-", "+": ".-.-.", "-": "-....-", "_": "..--.-",
    "\"": ".-..-.", "$": "...-..-", "@": ".--.-.", " ": "/"
}
MORSE_CODE_REVERSE = {v: k for k, v in MORSE_CODE.items()}


def base_64(text: str, decrypt: bool = False) -> str:
    base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    base64_index = {c: i for i, c in enumerate(base64_chars)}
    base64_padding = "="

    def base64_encode(data):
        encoded = ""
        remainder = len(data) % 3
        padding_len = 3 - remainder if remainder else 0
        for i in range(0, len(data) - remainder, 3):
            chunk = data[i:i + 3]
            int_value = (chunk[0] << 16) + (chunk[1] << 8) + chunk[2]
            for j in range(4):
                bits = (int_value >> (6 * (3 - j))) & 0x3F
                encoded += base64_chars[bits]
        if padding_len:
            chunk = data[-remainder:]
            binary = "".join([format(i, "08b") for i in chunk]) + "0" * (6 - len(chunk) * 8 % 6)
            encoded += "".join([base64_chars[int(binary[j:j + 6], 2)] for j in range(
                0, len(binary), 6)]) + base64_padding * (padding_len % 3)
        return encoded

    def base64_decode(encoded):
        encoded = encoded.replace("\n", "").replace(" ", "")
        decoded = b""
        if not all([c in base64_chars or c == base64_padding for c in encoded]):
            raise ContentError
        position = encoded.find(base64_padding)
        padding_len = len(encoded) - position if position != -1 else 0
        encoded = encoded.rstrip(base64_padding)
        if padding_len > 2 or base64_padding in encoded:
            raise ContentError
        idx = 0
        while idx < len(encoded):
            chunk = encoded[idx:idx + 4]
            int_value = 0
            for j in range(4):
                if j < len(chunk):
                    int_value = (int_value << 6) + base64_index[chunk[j]]
            decoded += (int_value >> 16 & 0xFF).to_bytes(1, "big")
            decoded += (int_value >> 8 & 0xFF).to_bytes(1, "big")
            decoded += (int_value & 0xFF).to_bytes(1, "big")
            idx += 4
        if padding_len:
            try:
                decoded = decoded[:-3] + int(format(int.from_bytes(
                    decoded[-3:], "big") >> 2 * padding_len, "b"), 2).to_bytes(3 - padding_len, "big")
            except OverflowError:
                raise ContentError
        return decoded

    if decrypt:
        decoded_bytes = base64_decode(text).decode("UTF-8")
        return decoded_bytes
    else:
        encoded_bytes = base64_encode(text.encode("UTF-8"))
        return encoded_bytes


def atbash(text: str) -> str:
    return "".join([chr(219 - ord(c)) if "a" <= c <= "z"
                    else chr(155 - ord(c)) if "A" <= c <= "Z" else c for c in text])


def rot13(text: str) -> str:
    return caesar(text, 13)


def caesar(text: str, shift: int, decrypt: bool = False) -> str:
    result = ""
    if decrypt:
        shift = -shift
    for char in text:
        if char.isalpha():
            ascii_offset = ORD_a if char.islower() else ORD_A
            new_char = chr((ord(char) - ascii_offset + shift) % 26 + ascii_offset)
            result += new_char
        else:
            result += char
    return result


def vigenere(text: str, key: str, decrypt: bool = False) -> str:
    result = ""
    key_index = 0
    for char in text:
        if char.isalpha():
            key_char = key[key_index]
            key_index = (key_index + 1) % len(key)
            shift = ord(key_char.lower()) - ORD_a
            if decrypt:
                shift = 26 - shift
            if char.isupper():
                result += chr((ord(char) - ORD_A + shift) % 26 + ORD_A)
            else:
                result += chr((ord(char) - ORD_a + shift) % 26 + ORD_a)
        else:
            result += char
    return result


def substitution(text: str, sub_map: dict, decrypt: bool = False) -> str:
    result = ""
    if decrypt:
        reverse_map = {v: k for k, v in sub_map.items()}
        sub_map = reverse_map
    for char in text:
        if char in sub_map:
            result += sub_map[char]
        else:
            result += char
    return result


def morse(text: str, decrypt: bool = False) -> str:
    code_dict = MORSE_CODE_REVERSE if decrypt else MORSE_CODE
    result = []
    if decrypt:
        text = text.split()
    else:
        text = text.upper()
    for item in text:
        if item in code_dict:
            result.append(code_dict[item])
        else:
            result.append(item)
    char = "" if decrypt else " "
    return char.join(result)


def playfair(text: str, key: str, decrypt: bool = False) -> str:
    def shift(coordinate, direction):
        row, col = coordinate
        if direction == "R":
            col = (col + 1) % 5
        elif direction == "L":
            col = (col - 1) % 5
        elif direction == "D":
            row = (row + 1) % 5
        elif direction == "U":
            row = (row - 1) % 5
        return row, col

    key = key.upper().replace("J", "I")
    dedup = []
    for k in key:
        if k.isalpha() and k not in dedup:
            dedup.append(k)
    key = "".join(dedup)
    if len(text) & 1 != 0:
        text += key[0]
    char_to_coord = {}
    coords = [(i, j) for i in range(5) for j in range(5)]
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
               "W", "X", "Y", "Z"]
    for char, coord in zip(key, coords):
        char_to_coord[char] = coord
        letters.remove(char)
    for char, coord in zip(letters, coords[len(key):]):
        char_to_coord[char] = coord
    coord_to_char = {v: k for k, v in char_to_coord.items()}
    result = ""
    for i in range(0, len(text), 2):
        pair = text[i:i + 2]
        is_lowercase1 = pair[0].islower()
        is_lowercase2 = pair[1].islower()
        c1, c2 = pair.upper().replace("J", "I")
        if c1 != c2 and pair.isalpha():
            c1_coord, c2_coord = char_to_coord[c1], char_to_coord[c2]
            if c1_coord[0] == c2_coord[0]:
                c1_coord = shift(c1_coord, "L") if decrypt else shift(c1_coord, "R")
                c2_coord = shift(c2_coord, "L") if decrypt else shift(c2_coord, "R")
            elif c1_coord[1] == c2_coord[1]:
                c1_coord = shift(c1_coord, "U") if decrypt else shift(c1_coord, "D")
                c2_coord = shift(c2_coord, "U") if decrypt else shift(c2_coord, "D")
            else:
                c1_coord, c2_coord = (c1_coord[0], c2_coord[1]), (c2_coord[0], c1_coord[1])
            encrypted_c1, encrypted_c2 = coord_to_char[c1_coord], coord_to_char[c2_coord]
            if is_lowercase1:
                encrypted_c1 = encrypted_c1.lower()
            if is_lowercase2:
                encrypted_c2 = encrypted_c2.lower()
            result += encrypted_c1 + encrypted_c2
        else:
            result += pair
    return result


def hill256(text: bytes, key: list, decrypt: bool = False) -> bytes:
    from .Array import fill
    from .Matrix import mat

    def decrypt_key(k):
        try:
            det = round(k.det())
        except ValueError:
            raise ValueError("The key must be a square matrix")
        res = 0
        while res < mod:
            if det * res % mod == 1:
                break
            else:
                res = res + 1
                if res == mod:
                    raise ValueError("The key square matrix does not have a multiplicative inverse element")
        return round(k.inv() * det % mod) * res % mod

    if not isinstance(text, bytes):
        raise TypeError("The input content needs to be encoded as a byte type")
    mod = 0x100
    length = len(text)
    key = mat(key)
    if decrypt:
        key = decrypt_key(key)
    cols = key.cols
    result = key @ mat(fill([cols, length // cols + bool(length % cols)], text, False)) % mod
    return b"".join([result[i][j].to_bytes(1, "big") for i in range(result.rows) for j in range(result.cols)])


def ksa(key: bytes) -> list:
    if not isinstance(key, bytes):
        raise TypeError("The key needs to be of byte type")
    length = len(key)
    if not length:
        raise ValueError("The key cannot be empty")
    s = list(range(0x100))
    j = 0
    for i in range(0x100):
        j = (j + s[i] + key[i % length]) % 0x100
        s[i], s[j] = s[j], s[i]
    return s


def prga(s: list):
    i = 0
    j = 0
    while True:
        i = (i + 1) % 0x100
        j = (j + s[i]) % 0x100
        s[i], s[j] = s[j], s[i]
        k = s[(s[i] + s[j]) % 0x100]
        yield k


def rc4(text: bytes, key: bytes) -> bytes:
    if not isinstance(text, bytes):
        raise TypeError("The input content needs to be encoded as a byte type")
    stream = prga(ksa(key))
    result = bytearray()
    for byte in text:
        k = next(stream)
        result.append(byte ^ k)
    return bytes(result)
