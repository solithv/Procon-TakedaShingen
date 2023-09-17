import requests as req
import os
import json
from dotenv import load_dotenv
class API:

    #.envはトークンを格納しているファイル
    load_dotenv(".env")
    token = os.getenv("TOKEN")
    match_url = os.getenv("MATCH_URL")
    header = {"procon-token":token}
    query = {"token":token}
    
    def get_match(self):
        """
        試合一覧取得API
        返り値: 試合idを格納した行列
        """
        r = req.get(self.match_url, headers=self.header)

        matches = r.json()
        id_ = []
        for match in matches["matches"]:
            id_.append(f'/{match["id"]}')
        return id_
    
    def get_field(self,path):
        """
        試合状態取得API
        引数(試合id)
        """
        r = req.get(f"{self.match_url}/{path}",headers=self.header)
        field = r.json()
        return field

    def post_actions(self,act,path):
        """
        行動計画更新API
        引数(jsonファイル , 試合id)
        """

        req.post(f"{self.match_url}/{path}", headers=self.header, json=act)

        