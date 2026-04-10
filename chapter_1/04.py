text = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
alphabet_text = text.replace(",", "").replace(".", "").split() #文字列からカンマとピリオドを取り除いてスペースで分割してリストにする
dictionary = {}
for i in range(len(alphabet_text)):
    if i + 1 in [1, 5, 6, 7, 8, 9, 15, 16, 19]: # i+1が1, 5, 6, 7, 8, 9, 15, 16, 19のときは、単語の最初の文字をキーにする
        dictionary[alphabet_text[i][0]] = i + 1
    else: #それ以外のときは、単語の最初の2文字をキーにする
        dictionary[alphabet_text[i][0:2]] = i + 1
print(dictionary)
