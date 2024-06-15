## Slack Botによる雑談対話システムの作成
### 概要
- Slackを利用した対話型botを作成した。
- SVMを利用したネガポジ判定による返答や3つの対話戦略「キーワード照合ルール」「発話行為タイプ」「定型文」を実装した。
- また、自分で考えた天気戦略と容量戦略の2つを実装した。
- 天気機能では、第1回に学んだBeautifulSoupを利用しYahoo天気からデータを取得し現在の天気を返答することを目的とした。
- 容量機能では、Psutilモジュールを使用して使用者のディスク容量を返答することを目的とした。

### 1.目的
- 対話型チャットbotを作成し、実装した各戦略を評価する。
- 「天気を教えて」のキーワードをもとに、リアルタイムでYahoo！天気からスクレイピングし、現在と次の日の天気を返答する。
- 「ディスク容量を教えて」のキーワードを取得すると自分のpcの容量を教えてくれるような機能を実装する。
- 講義で習ったことを活かした機能を実装することを目的とした。

### プログラム解説
#### 3つの対話戦略のプログラム解説
- 「対話の生成」のプログラムを実行すると、キーワード照合ルールを初期化する関数①stepKeywordMatchingRuleが実行される。キーワード照合ルールでは、kw_matching_rule.txtを読み込み、「キーワードと応答文」のペアを作り②kRuleListに追加する。
- 応答文を生成する関数③generateResponseがinputを渡して実行される。応答候補リスト④candidateListを空にする。
- inputをJanomeで形態素解析し、単語（Wordクラス）をwordlistに代入する。
- キーワード照合ルールを利用した応答を生成する関数⑤grnerateResponseByRuleがinputを引数として実行される。
- inputにリストkRuleList内のキーワードが含まれていたら，リストcandidateListに「応答文とスコア」のペアを追加する
- ユーザ入力文に含まれる名詞を利用した応答文を生成する関数⑥generateResponseBylnputTopicがwordlistを引数として実行される。
- wordlistのうち名詞について，「は好きですか？」か「って何ですか？」のどちらかを後ろに付けて応答文を作成し，リスト④に「応答文とスコア」のペアを追加する。
- 無難な応答を返す関数⑦generataOtherResponseが実行される。
- 「なるほど」か「それで」を応答文として，リスト④に「応答文とスコア」のペアを追加する。
- 変数retに「デフォルト」を，変数⑧maxSoreに-1.0を代入する。
- リスト④からfor文で一つずつ「応答文とスコア」のペアを取り出し，スコアが変数⑧より大きかったら変数retをこの応答文に更新し，変数⑧をこのスコアに更新する。
- 最もスコアの高かった応答文retが戻り値となる。戻り値がoutputに代入される。
- さらに変数inputに「私は学生です」と代入して、上記の同様の処理により応答文を出力する。
#### 天気機能
- Yahoo天気から現在の天気を取得する関数（YahooWeather）を作成する。RequestsとBeautifulSoupをインポートする。
- Getメソッドを使用しURLにリクエストしてレスポンスを取得する。
- 取得したレスポンスから BeautifulSoup オブジェクトを作る。
- BeautifulSoupのfindメソッドを使用しリストで返す。
- 必要な要素のインデックスを指定する(日付はrs[0],天気はrs[1],明日の天気はrs[19])
- If文で「天気をおしえて」を取得したときにYahooWeatherを呼び出す。
#### 容量機能
- If文で「ディスク容量を教えて」と入力された際に処理を行う。
- psutilメソッドを使用しディスクの容量を取得しoutputに値を代入する。
#### ネガポジ判定のプログラム
- ユーザ入力をSVMで感情推定して返答
- ポジティブなら「よかったね」 ネガティブなら「ざんねん」 と返すようにする
- ネガポジ判定の学習には時間がかかるので，第3回（演習3-2）で学習 したSVMの学習済みモデルをファイルに保存したものを読み込む。
- モジュールpickle：オブジェクトをバイト列に変換してファイルに保存。
- 保存した svmclassifier.pkl をフォルダにコピーし，pickle.loadで読み込む。
- 単語リストを読み込み，リストbasicFormListに追加。
##### 関数の作成
  - 第3回演習で使用した関数の再利用
  - makeCounter：頻度のCounterを作成する
  - makeVectorList：複数の文のCounterリストと単語リストから、ベクトルのリストを作成する。
##### 新しい関数の作成
- negaposiAnalyzer (text, classifier, basicFormList) ：ネガポジ判定して，出力結果 を返す
##### ＜引数＞
  - text:入力文
  - classifier：学習済みSVMモデル
  - basicFormList：ベクトル化に使用する単語リスト
##### ネガポジ判定の結果を返す関数
- negaposiAnalyzer(text, classifier, basicFormList)の処理の流れ
1. textを形態素解析 
2. 単語のCounterを作成 
3. CounterをcounterListに追加（classifierが予測する際に，引数がベクトルのリス トである必要があるため，1文でもCounterをリストに追加） 
4. counterList内のCounterをベクトル化（basicFormListの単語を使用） 
5. ネガポジ判定
6. 予測結果によって出力を決定して返す 38 判定結果が「1」だったら「よかったね」 それ以外だったら「ざんねん」を返す
- 出力生成部分では，ネガポジ判定の結果を返す関数で作成 した応答文をSlackに戻すようになっている

