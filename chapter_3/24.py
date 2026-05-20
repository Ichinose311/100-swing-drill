import pandas as pd
import re

df = pd.read_json('jawiki-country.json.gz', lines=True) #JSON形式のファイルを読み込む。lines=Trueは、各行が独立したJSONオブジェクトであることを示す。
uk_text = df.query('title=="イギリス"')["text"].values[0] #title行がイギリスの行を抽出し、text列の値を取り出す。最初の要素を取り出す。
# re.fiandall(パターン,文字列) マッチした部分をリストで返す
# 正規表現
# \[\[ : [[Wikiのリンクの開始
# (ファイル|File) : ファイルまたはFileのどちらかにマッチ
# : : コロン
# ([^]|]+?) : |または]以外の文字が1回以上   
# (\|.*?)+ : |に続く任意の文字が0回以上
# \]\] : Wikiのリンクの終了
for file in re.findall(r"\[\[(ファイル|File):([^]|]+?)(\|.*?)+\]\]", uk_text):
    # findallの戻り値は、マッチした部分のグループごとのタプルのリスト
    '''[
            ('ファイル', 'Example.jpg', '|thumb|200px'),
            ('File', 'Image.png', '|right')
        ]'''
    # file = ('ファイル', 'Example.jpg', '|thumb') インデックス1のとき
    # file[1] = 'Example.jpg'
    print(file[1])
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
'''
