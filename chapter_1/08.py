def cipher(text):
    ciphered_text = ""
    for char in text:
        if char.islower(): #小文字の場合
            ciphered_text += chr(219 - ord(char)) #219から文字コードを引いた文字を追加
        else: #それ以外の文字はそのまま追加
            ciphered_text += char
    return ciphered_text
print(cipher("Hello, World!")) #暗号化
print(cipher(cipher("Hello, World!"))) #復号化
