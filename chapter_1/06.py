text_1 = "paraparaparadise"
text_2 = "paragraph"
def n_gram(n, text):
    text = text.replace(" ", "")
    return [text[i:i+n] for i in range(len(text)-n+1)]
x = set(n_gram(2, text_1)) #リストから集合(要素は重複しない)に変換
y = set(n_gram(2, text_2))
print(x | y) #xとyの和集合
print(x & y) #xとyの積集合
print(x - y) #xからyを引いた集合
print('se' in x or 'se' in y) #'se'というbi-gramがxかyのどちらかに含まれているか
