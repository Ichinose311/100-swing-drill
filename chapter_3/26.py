import pandas as pd
import re
def remove_emphasis(dc):
    r = re.compile("'+") 
    # 'が1回以上続く部分にマッチする正規表現
    return {k: r.sub("", v) for k, v in dc.items()} 
    # 辞書の各値に対して、正規表現rを使って、'を空文字に置換する。辞書内包表記を使って、新しい辞書を作成する。

df = pd.read_json('jawiki-country.json.gz', lines=True) #JSON形式のファイルを読み込む。lines=Trueは、各行が独立したJSONオブジェクトであることを示す。
uk_text = df.query('title=="イギリス"')["text"].values[0] #title行がイギリスの行を抽出し、text列の値を取り出す。最初の要素を取り出す。
uk_texts = uk_text.split("\n")
# \| : ｜からはじまる行を対象にする
# (.+?) : 1文字以上の任意の文字を非貪欲マッチで取得する
# \s=\s* : =の前後に0個以上の空白があることを許容する
# (.+) : 1文字以上の任意の文字を貪欲マッチで取得する
# |国名 = イギリス みたいな行からキー:国名、値:イギリスを抽出するための正規表現
pattern = re.compile("\|(.+?)\s=\s*(.+)")
result = {}
for line in uk_texts:
    r = re.search(pattern, line)
    if r:
        result[r[1]] = r[2]
print(remove_emphasis(result))
#出力結果
'''
{0}}1日', '確立形態3': '[[グレートブリテン及びアイルランド連合王国]]成立<br />
（[[合同法 (1800年)|1800年合同法]]）', '確立年月日3': '1801年{{0}}1月{{0}}1
日', '確立形態4': '現在の国号「グレートブリテン及び北アイルランド連合王国」に変更',
 '確立年月日4': '1927年{{0}}4月12日', '通貨': '[[スターリング・ポンド|UKポンド]
 ] (£)', '通貨コード': 'GBP', '時間帯': '±0', '夏時間': '+1', 'ISO 3166-1':
   'GB / GBR', 'ccTLD': '[[.uk]] / [[.gb]]<ref>使用は.ukに比べ圧倒的少数。</
   ref>', '国際電話番号': '44', '注記': '<references/>'}
'''