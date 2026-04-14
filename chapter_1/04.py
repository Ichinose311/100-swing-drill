# 意味のある名前を使う（p.18）
text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

words = text.replace(",", "").replace(".", "").split()
element_dict = {}

# 定数はまとめる（p.72）
special_indices = {1, 5, 6, 7, 8, 9, 15, 16, 19}

for i, word in enumerate(words, start=1):
    if i in special_indices:
        element_dict[word[0]] = i
    else:
        element_dict[word[:2]] = i

print(element_dict)
