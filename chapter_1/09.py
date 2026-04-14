import random

# 意味のある名前を使う（p.18）
text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

words = text.split()
result_text = ""

for word in words:
    if len(word) > 4:
        first_char = word[0]
        last_char = word[-1]
        middle = word[1:-1]

        # 中間変数で処理を分かりやすくする（p.53）
        shuffled_middle = ''.join(random.sample(middle, len(middle)))

        result_text += first_char + shuffled_middle + last_char + " "
    else:
        result_text += word + " "

print(result_text)