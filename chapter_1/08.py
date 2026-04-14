# 変数名で役割を明確にする（p.18）
def cipher(text):
    result = ""

    for char in text:
        if char.islower():
            # 複雑な式には説明を付ける（p.56）
            result += chr(219 - ord(char))
        else:
            result += char

    return result

print(cipher("Hello, World!")) 
print(cipher(cipher("Hello, World!")))