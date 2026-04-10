text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
alphabet_text = text.replace(",", "").replace(".", "") #文字列からカンマとピリオドを取り除く
word_list = alphabet_text.split() #文字列をスペースで分割してリストにする
word_length_list = [len(word) for word in word_list] #リスト内包表記を使って、各単語の文字数をリストにする
print(word_length_list)