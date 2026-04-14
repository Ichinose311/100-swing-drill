# 明確な単語を選ぶ（p.10）
text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

words = text.replace(",", "").replace(".", "").split()
element_dict = {}

# 「名前付き引数」コメント（p.77）
special_indices = {1, 5, 6, 7, 8, 9, 15, 16, 19} #特定の単語のインデックスを指定するためのセット

for i, word in enumerate(words, start=1):
    if i in special_indices:
        element_dict[word[0]] = i
    else:
        element_dict[word[:2]] = i

print(element_dict)
