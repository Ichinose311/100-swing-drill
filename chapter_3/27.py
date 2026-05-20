import pandas as pd
import re
def remove_emphasis(dc):
    r = re.compile("'+") # 'が1回以上続く部分にマッチする正規表現
    return {k: r.sub("", v) for k, v in dc.items()} # 辞書の各値に対して、正規表現rを使って、'を空文字に置換する。辞書内包表記を使って、新しい辞書を作成する。

def remove_internal_link(dc):
    r = re.compile(r"\[\[([^|]+?)(\|.*)?\]\]") # [[で始まり、|または]]で終わる部分にマッチする正規表現
    return {k: r.sub(r"\1", v) for k, v in dc.items()} # 辞書の各値に対して、正規表現rを使って、[[と]]を削除し、|以降の文字列も削除する。辞書内包表記を使って、新しい辞書を作成する。

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
print(remove_internal_link(remove_emphasis(result)))
#出力結果
'''
{'略名 ': 'イギリス', '日本語国名': 'グレートブリテン及び北アイルランド連合王国', 
'公式国名': '{{lang|en|United Kingdom of Great Britain and Northern Ire
land}}<ref>英語以外での正式国名:<br />', '国旗画像': 'Flag of the United Ki
ngdom.svg', '国章画像': 'ファイル:Royal Coat of Arms of the United Kingdo
m.svg', '国章リンク': '（イギリスの国章）', '標語': '{{lang|fr|Dieu et mon d
roit}}<br />（フランス語:Dieu et mon droit）', '国歌': '女王陛下万歳}}', '地
図画像': 'Europe-UK.svg', '位置画像': 'United Kingdom (+overseas territor
ies) in the World (+Antarctica claims).svg', '公用語': '英語', '首都': 'ロ
ンドン（事実上）', '最大都市': 'ロンドン', '元首等肩書': 'イギリスの君主', '元首等
氏名': 'エリザベス2世', '首相等肩書': 'イギリスの首相', '首相等氏名': 'ボリス・ジョ
ンソン', '他元首等肩書1': '貴族院 (イギリス)', '他元首等氏名1': ':en:Norman Fowl
er, Baron Fowler', '他元首等肩書2': '庶民院 (イギリス)', '他元首等氏名2': '{{仮
リンク|リンゼイ・ホイル|en|Lindsay Hoyle}}', '他元首等肩書3': '連合王国最高裁判所'
, '他元首等氏名3': ':en:Brenda Hale, Baroness Hale of Richmond', '面積順位': 
'76', '面積大きさ': '1 E11', '面積値': '244,820', '水面積率': '1.3%', '人口統計
年': '2018', '人口順位': '22', '人口大きさ': '1 E7', '人口値': '6643万5600<ref>
{{Cite web|url=https://www.ons.gov.uk/peoplepopulationandcommunity/populati
onandmigration/populationestimates|title=Population estimates - Office for 
National Statistics|accessdate=2019-06-26|date=2019-06-26}}</ref>', '人口密度
値': '271', 'GDP統計年元': '2012', 'GDP値元': '1兆5478億<ref name="imf-statisti
cs-gdp">[http://www.imf.org/external/pubs/ft/weo/2012/02/weodata/weorept.aspx
?pr.x=70&pr.y=13&sy=2010&ey=2012&scsm=1&ssd=1&sort=country&ds=.&br=1&c=112&s=
NGDP%2CNGDPD%2CPPPGDP%2CPPPPC&grp=0&a=IMF>Data and Statistics>World Economic 
Outlook Databases>By Countrise>United Kingdom]</ref>', 'GDP統計年MER': '2012',
 'GDP順位MER': '6', 'GDP値MER': '2兆4337億<ref name="imf-statistics-gdp" />', '
 GDP統計年': '2012', 'GDP順位': '6', 'GDP値': '2兆3162億<ref name="imf-statistic
 s-gdp" />', 'GDP/人': '36,727<ref name="imf-statistics-gdp" />', '建国形態': '
 建国', '確立形態1': 'イングランド王国／スコットランド王国<br />（両国とも合同法 (1707年)
 まで）', '確立年月日1': '927年／843年', '確立形態2': 'グレートブリテン王国成立<br />（1
 707年合同法）', '確立年月日2': '1707年{{0}}5月{{0}}1日', '確立形態3': 'グレートブリテン
 及びアイルランド連合王国成立<br />（合同法 (1800年)）', '確立年月日3': '1801年{{0}}1月{
 {0}}1日', '確立形態4': '現在の国号「グレートブリテン及び北アイルランド連合王国」に変更', '確
 立年月日4': '1927年{{0}}4月12日', '通貨': 'スターリング・ポンド (£)', '通貨コード': 'GB
 P', '時間帯': '±0', '夏時間': '+1', 'ISO 3166-1': 'GB / GBR', 'ccTLD': '.uk / .gb
 <ref>使用は.ukに比べ圧倒的少数。</ref>', '国際電話番号': '44', '注記': '<references/>'}
'''