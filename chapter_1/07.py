# 名前のフォーマットで情報を伝える（p.25）
def create_template_sentence(time, subject, value):
    return f"{time}時の{subject}は{value}"

print(create_template_sentence(12, "気温", 22.4))
