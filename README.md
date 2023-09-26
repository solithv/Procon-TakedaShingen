# 武田進言

高専プロコン第34回福井大会 競技部門のリポジトリ

## How to use

### 準備

```bash
pip install -r requirements.txt
```

`.env`ファイルを作成

```txt:.env
TOKEN=***
MATCH_URL=http://***
```
`TOKEN`: 接続トークン \
`MATCH_URL`: サーバURL

### 学習用データ作成

```bash
python annotator.py
```

### 学習

```bash
python train.py
```

### ローカルプレイ

```bash
python app.py
```

### サーバでの動作確認
サーバ用configファイル作成
```bash
python make_server_configs.py
```

サーバを動作させて以下をそれぞれ実行

```bash
python main.py
python dummy.py
```

## フォルダ説明

### ./assets

GUIに使用する画像

### ./dataset

学習用データ

### ./field_data

マップ情報

### ./model

学習済みモデル

### ./MyEnv

ゲーム環境実装

### ./NN

機械学習関係実装

### ./Utils

API通信、その他機能実装
