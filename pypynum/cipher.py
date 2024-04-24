def dna(string: str, decrypt: bool = False) -> str:
    if not isinstance(decrypt, bool):
        raise TypeError("The parameter 'decrypt' can only be a Boolean value")
    if decrypt is False:
        relation = {0: "A", 1: "C", 2: "G", 3: "T"}
        result = ""
        data = string.encode()
        for byte in data:
            qua = ""
            while byte > 0:
                byte, rem = divmod(byte, 4)
                qua = relation[rem] + qua
            result += relation[0] * (4 - len(qua)) + qua
        return "".join(sum(list(zip(result[:len(result) // 2], result[len(result) // 2:][::-1])), ()))
    else:
        relation = {"A": "0", "C": "1", "G": "2", "T": "3"}
        result = b""
        data = string[::2] + string[::-2]
        for item in relation:
            data = data.replace(item, relation[item])
        for item in range(0, len(data), 4):
            result += int(data[item:item + 4], 4).to_bytes(1, "big")
        return result.decode()
