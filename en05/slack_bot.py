from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import settings
from janome.tokenizer import Tokenizer

import logging
import re
from typing import Callable

from slack_bolt import App, Say, BoltContext
from slack_sdk import WebClient

app = App(token=settings.SLACK_BOT_TOKEN)

# 単語のクラス
class Word:
    def __init__(self, token):
        # 表層形
        self.text = token.surface

        # 原型
        self.basicForm = token.base_form

        # 品詞
        self.pos = token.part_of_speech
        
    # 単語の情報を「表層系\t原型\t品詞」で返す
    def wordInfo(self):
        return self.text + "\t" + self.basicForm + "\t" + self.pos

# 引数のtextをJanomeで解析して単語リストを返す関数
def janomeAnalyzer(text):
    # 形態素解析
    t = Tokenizer()
    tokens = t.tokenize(text) 

    # 解析結果を1行ずつ取得してリストに追加
    wordlist = []
    for token in tokens:
        word = Word(token)
        wordlist.append(word)
    return wordlist

# 入力データからuseridとtextを取得
def body_parser(body):
    # useridの取得
    userid = body["event"]["user"]

    # textの取得
    blocks = body["event"]["blocks"][0]
    elements = blocks["elements"][0]
    texts = elements["elements"][1]
    text = texts["text"]

    return userid, text.strip()

import pickle

# 保存したモデルをロードする
filename = "svmclassifier.pkl"
loaded_classifier = pickle.load(open(filename, "rb"))

# 単語リストを読み込みリストに保存
basicFormList = []
bffile = "basicFormList.txt"
for line in open(bffile, "r", encoding="utf_8"):
    basicFormList.append(line.strip())
print(len(basicFormList))

import random

# キーワード照合ルールのリスト（keywordMatchingRuleオブジェクトのリスト）
kRuleList = []

# 応答候補のリスト（ResponseCandidateオブジェクトのリスト）
candidateList = []

# キーワード照合ルールのクラス（キーワードと応答の組み合わせ）
class KeywordMatchingRule:
    def __init__(self, keyword, response):
        self.keyword = keyword
        self.response = response

# 応答候補のクラス（応答候補とスコアの組み合わせ）
class ResponseCandidate:
    def __init__(self, response, score):
        self.response = response
        self.score = score
    def print(self):
        print("候補文 [%s, %.5f]" % (self.response, self.score))

# キーワード照合ルールを初期化する関数
def setupKeywordMatchingRule():
    kRuleList.clear()
    for line in open('kw_matching_rule.txt', 'r', encoding="utf_8"):
        arr = line.split(",")    
        # keywordMatchingRuleオブジェクトを作成してkRuleListに追加
        kRuleList.append(KeywordMatchingRule(arr[0], arr[1].strip()))
        
# キーワード照合ルールを利用した応答候補を生成する関数
def generateResponseByRule(inputText):
    for rule in kRuleList:
        # ルールのキーワードが入力テキストに含まれていたら
        if(rule.keyword in inputText):
            # キーワードに対応する応答文とスコアでResponseCandidateオブジェクトを作成してcandidateListに追加
            cdd = ResponseCandidate(rule.response, 1.0 + random.random())
            candidateList.append(cdd)

# ユーザ入力文に含まれる名詞を利用した応答候補を生成する関数
def generateResponseByInputTopic(inputWordList):
    # 名詞につなげる語句のリスト
    textList = ["は好きですか？", "って何ですか？", "っていいよね"]
    
    for w in inputWordList:
        pos2 = w.pos.split(",")
        # 品詞が名詞だったら
        if pos2[0]=='名詞':
            cdd = ResponseCandidate(w.basicForm + random.choice(textList), 
                                    1.0 + random.random())
            candidateList.append(cdd)
            
# 無難な応答を返す関数
def generateOtherResponse():
    # 無難な応答のリスト
    bunanList = ["なるほど", "それで？", "ふむふむ"]

    # ランダムにどれかをcandidateListに追加
    cdd = ResponseCandidate(random.choice(bunanList), 0.5 + random.random())
    candidateList.append(cdd)

from collections import Counter

# 単語情報リストを渡すとカウンターを返す関数
def makeCounter(wordList):
    basicFormList = []
    for word in wordList:
        basicFormList.append(word.basicForm)
    # 単語の原型のカウンターを作成
    counter = Counter(basicFormList)
    return counter

# Counterのリストと単語リストからベクトルのリストを作成する関数
def makeVectorList(counterList, basicFormList):
    vectorList = []
    for counter in counterList:
        vector = []
        for word in basicFormList:
            vector.append(counter[word])
        vectorList.append(vector)
    return vectorList

from sklearn import svm

# ネガポジ判定の結果を返す関数
# 引数 text:入力文, classifier：学習済みモデル, basicFormList：ベクトル化に使用する単語リスト
def negaposiAnalyzer(text, classifier, basicFormList):
    # 形態素解析して頻度のCounterを作成
    counterList = []
    wordlist = janomeAnalyzer(text)
    counter = makeCounter(wordlist)
    
    # 1文のcounterだが，counterListに追加
    counterList.append(counter)

    # Counterリストと単語リストからベクトルのリストを作成
    vectorList = makeVectorList(counterList, basicFormList)

    # ベクトルのリストに対してネガポジ判定
    predict_label = classifier.predict(vectorList)

    # 入力文のベクトル化に使用された単語を出力
    for vector in vectorList:
        wl=[]
        for i, num in enumerate(vector):
            if(num==1):
                wl.append(basicFormList[i])
        print(wl)

    # 予測結果を出力
    print(predict_label)

    # 予測結果によって出力を決定
    if predict_label[0]=="1":
        output = "よかったね"
    else:
        output = "ざんねん"
        
    return output

def generateNegaposiResponse(inputText):
    # ネガポジ判定を実行
    output = negaposiAnalyzer(inputText, loaded_classifier, 
                              basicFormList)
    
    # 応答候補に追加
    cdd = ResponseCandidate(output, 0.7 + random.random())
    candidateList.append(cdd)     

#追加---------------------------------------------
def YahooWeather():
    import requests
    from bs4 import BeautifulSoup
    url = "https://weather.yahoo.co.jp/weather/jp/13/4410.html"
    
    # URLにリクエストしてレスポンスを取得
    res = requests.get(url)
    
    # 取得したレスポンスから BeautifulSoup オブジェクトを作る
    soup = BeautifulSoup(res.text, 'html.parser')
    
    rs = soup.find(class_='forecastCity')
    rs = [i.strip() for i in rs.text.splitlines()]
    rs = [i for i in rs if i != ""]
    return "東京都" + rs[0] + "の天気は" + rs[1] + "、明日の天気は" + rs[19] + "です。"
#--------------------------------------------------

# 応答文を生成する関数
def generateResponse(inputText):
    
    # 出力文
    output = ""
    
    # 決まったキーワードを含むとき（タスク指向）のときの出力
    #入力文が「時間を教えて」を含んでいたら時間を出力に設定
    if "時間を教えて" in inputText:
        import datetime
        dt_now = datetime.datetime.now()
        dt_now_str = dt_now.strftime("%Y/%m/%d %H:%M")
        output = dt_now_str
        
# 追加 ------------------------------------
    elif "ディスク容量を教えて" in inputText:
        import psutil
        #ディスク容量を取得
        dsk = psutil.disk_usage('/')
        output = str(round(dsk.total / 1073741824)) + "GB" + "\n" + str(round(dsk.free/1073741824)) + "GB空いています"
        
    elif "夕飯" in inputText:
        movie_list = ["ハンバーグ","オムライス","肉じゃが"]
        output = random.choice(movie_list)
        
    elif "天気を教えて" in inputText:
        output = YahooWeather()
#------------------------------------------        
           
    #それ以外は非タスク指向（雑談対話）で返す
    else:
   
        # 応答文候補を空にしておく
        candidateList.clear()

        # 形態素解析した後，3つの戦略を順番に実行
        wordlist = janomeAnalyzer(inputText)
        generateResponseByRule(inputText)
        generateResponseByInputTopic(wordlist)
        generateOtherResponse()
        

        # ネガポジ判定の結果を応答候補に追加
        generateNegaposiResponse(inputText)

        ret="デフォルト"
        maxScore=-1.0

        # scoreが最も高い応答文候補を戻す
        for cdd in candidateList:
            cdd.print()
            if cdd.score > maxScore:
                ret=cdd.response
                maxScore = cdd.score
        output = ret
        
    return output

# キーワードマッチングルールの読み込み
setupKeywordMatchingRule()

# メンションされたときに返答する
@app.event("app_mention")
def event_mention(context, body, say, logger):
    logger.info(body)
    
    # 入力からユーザIDとテキストを取得
    userid, text = body_parser(body)

    # システムの出力を生成
    output = generateResponse(text)
    
    channel=body["event"]["channel"]
    timestamp=body["event"]["ts"]
    
      #リアクションを返す
    if "かっこいい" in text:
        context.client.reactions_add(
            name="heart",
            channel=channel,
            timestamp=timestamp,
        )
    
    #画像を投稿する
    url = "http://1.bp.blogspot.com/-Hli6afTS54w/UgSMCMQNnNI/AAAAAAAAW6c/FXu-zvewMWY/s180-c/food_hamburg.png"
    if "ハンバーグ" in text:
        output = f"ハンバーグ {url}" 
        
    #アップロードしたファイルのURLを取得する
    if "files" in body["event"]:
        fileinfo = body["event"]["files"][0]
        file_url = fileinfo["permalink"]
        output = f"ファイルを受け取りました {file_url}"
    
    # Slackで返答
    say(f"<@{userid}> {output}")
    
@app.event("message")
def handle_message_events(body, logger):
    logger.info(body)
    
if __name__ == "__main__":
    handler = SocketModeHandler(app, settings.SLACK_APP_TOKEN)
    handler.start()