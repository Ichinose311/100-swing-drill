import random

# 明確な単語を選ぶ（p.10）
text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

words = text.split()
result_text = ""

for word in words:
    if len(word) > 4:
        # コードを「段落」に分割する (p.51)
        # 説明変数(p.100)
        first_char = word[0] #範囲を指定するときはfirstとlastを使う（p.32）
        last_char = word[-1]
        middle = word[1:-1]

        # 中間変数で処理を分かりやすくする（p.53）
        shuffled_middle = ''.join(random.sample(middle, len(middle)))

        result_text += first_char + shuffled_middle + last_char + " "
    else:
        result_text += word + " "

print(result_text)