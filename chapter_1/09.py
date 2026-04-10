import random

text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
print_text = ""
text_list = text.split() 
for word in text_list:
    if len(word) > 4:
        first = word[:1] #単語の最初の文字
        last = word[-1:] #単語の最後の文字
        other = word[1:-1] #単語の最初と最後の文字以外の部分
        shuffled_other = ''.join(random.sample(other, len(other)))  # otherの文字をランダムに入れ替え
        print_text += first + shuffled_other + last + " "
    else:
        print_text += word + " "
print(print_text)
