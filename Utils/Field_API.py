import os
import time

import requests as req
from dotenv import load_dotenv


class API:
    # .envはトークンを格納しているファイル
    load_dotenv(".env")
    token = os.getenv("TOKEN")
    match_url = f"{os.getenv('MATCH_URL')}/matches"
    header = {"procon-token": token}

    def get_match(self):
        """
        試合一覧取得API
        返り値 id_: 試合idを格納した行列
        """
        while True:
            # 返答が200(正常)でなければ0.1
            r = req.get(self.match_url, headers=self.header)
            if r.status_code == 200:
                break
            print("get_match", r.status_code, r.text)
            time.sleep(0.1)

        matches = r.json()
        return matches["matches"]

    def get_field(self, path):
        """
        試合状態取得API
        引数(試合id)
        """
        while True:
            r = req.get(f"{self.match_url}/{path}", headers=self.header)
            if r.status_code == 200:
                break
            print("get_field", r.status_code, r.text)
            time.sleep(0.1)

        field = r.json()
        return field

    def post_actions(self, act, path, opponent=False):
        """
        行動計画更新API
        引数(jsonファイル , 試合id)
        """
        header = self.header if not opponent else {"procon-token": "dummy-token"}
        while True:
            r = req.post(f"{self.match_url}/{path}", headers=header, json=act)
            if r.status_code == 200:
                break
            print("post_actions", r.status_code, r.text)
            time.sleep(0.1)
