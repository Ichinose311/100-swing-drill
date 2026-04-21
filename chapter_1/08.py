# 明確な単語を選ぶ（p.10）
def cipher(text):
    result = ""

    for char in text:
        if char.islower():
            # コードの意図を書く（p.76）
            # アルファベットの文字コードの合計219（a=97, z=122）から、現在の文字コードを引いた値
            result += chr(219 - ord(char))
        else:
            result += char

    return result

print(cipher("Hello, World!")) 
print(cipher(cipher("Hello, World!")))