import pandas as pd
import re

df = pd.read_json('jawiki-country.json.gz', lines=True) #JSON形式のファイルを読み込む。lines=Trueは、各行が独立したJSONオブジェクトであることを示す。
uk_text = df.query('title=="イギリス"')["text"].values[0] #title行がイギリスの行を抽出し、text列の値を取り出す。最初の要素を取り出す。
uk_texts = uk_text.split("\n")
# 基礎情報テンプレートの「|項目名 = 値」の行を探すための正規表現を作る
# \| : ｜からはじまる行を対象にする
# (.+?) : 1文字以上の任意の文字を非貪欲マッチで取得する
# \s=\s* : =の前後に0個以上の空白があることを許容する
# (.+) : 1文字以上の任意の文字を貪欲マッチで取得する
# |国名 = イギリス みたいな行からキー:国名、値:イギリスを抽出するための正規表現
pattern = re.compile("\|(.+?)\s=\s*(.+)")
# 取り出した項目名と値を入れるための空の辞書を作る
result = {}
for line in uk_texts:
    # 現在の1行が、「|項目名 = 値」の形になっているか調べる
    r = re.search(pattern, line)
    if r:
        # r[1] には項目名が入っている
        # r[2] には値が入っている
        # 例: 「|首都 = ロンドン」なら result["首都"] = "ロンドン"
        result[r[1]] = r[2]

print(result)
#出力結果
'''
Royal Coat of Arms of the United Kingdom.svg
Descriptio Prime Tabulae Europae.jpg
Lenepveu, Jeanne d'Arc au siège d'Orléans.jpg
London.bankofengland.arp.jpg
Battle of Waterloo 1815.PNG
Uk topo en.jpg
BenNevis2005.jpg
Population density UK 2011 census.png
2019 Greenwich Peninsula & Canary Wharf.jpg
Birmingham Skyline from Edgbaston Cricket Ground crop.jpg
Leeds CBD at night.jpg
Glasgow and the Clyde from the air (geograph 4665720).jpg
Palace of Westminster, London - Feb 2007.jpg
Scotland Parliament Holyrood.jpg
Donald Trump and Theresa May (33998675310) (cropped).jpg
Soldiers Trooping the Colour, 16th June 2007.jpg
City of London skyline from London City Hall - Oct 2008.jpg
Oil platform in the North SeaPros.jpg
Eurostar at St Pancras Jan 2008.jpg
Heathrow Terminal 5C Iwelumo-1.jpg
Airbus A380-841 G-XLEB British Airways (10424102995).jpg
UKpop.svg
Anglospeak.svg
Royal Aberdeen Children's Hospital.jpg
CHANDOS3.jpg
The Fabs.JPG
Wembley Stadium, illuminated.jpg
ichinoseyuuta@ichinoseyuutanoMacBook-Pro chapter_3 % uv run python 25.py
/Users/ichinoseyuuta/Inaba_lab/100-swing-drill/chapter_3/25.py:12: SyntaxWarning: invalid escape sequence '\|'
  pattern = re.compile("\|(.+?)\s=\s*(.+)")
{'略名 ': 'イギリス', '日本語国名': 'グレートブリテン及び北アイルランド連合王国', '公式国名': '{{lang|en|United Kingdom of Great 
Britain and Northern Ireland}}<ref>英語以外での正式国名:<br />', '国旗画像': 'Flag of the United Kingdom.svg', '国章画像': 
'[[ファイル:Royal Coat of Arms of the United Kingdom.svg|85px|イギリスの国章]]', '国章リンク': '（[[イギリスの国章|国章]]）', 
'標語': '{{lang|fr|[[Dieu et mon droit]]}}<br />（[[フランス語]]:[[Dieu et mon droit|神と我が権利]]）', '国歌': "[[女王陛下
万歳|{{lang|en|God Save the Queen}}]]{{en icon}}<br />''神よ女王を護り賜え''<br />{{center|[[ファイル:United States Navy 
Band - God Save the Queen.ogg]]}}", '地図画像': 'Europe-UK.svg', '位置画像': 'United Kingdom (+overseas territories) in
 the World (+Antarctica claims).svg', '公用語': '[[英語]]', '首都': '[[ロンドン]]（事実上）', '最大都市': 'ロンドン', '元首等肩
 書': '[[イギリスの君主|女王]]', '元首等氏名': '[[エリザベス2世]]', '首相等肩書': '[[イギリスの首相|首相]]', '首相等氏名': '[[ボリス・
 ジョンソン]]', '他元首等肩書1': '[[貴族院 (イギリス)|貴族院議長]]',('確立年月日4': '1927年{{0}}4月12日', '通貨': '[[スターリング・ポ
 ンド|UKポンド]] (£)', '
'''
