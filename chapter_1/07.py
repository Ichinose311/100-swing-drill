# 関数名は何をするか明確にする（p.18）
def create_template_sentence(time, subject, value):
    return f"{time}時の{subject}は{value}"

print(create_template_sentence(12, "気温", 22.4))
