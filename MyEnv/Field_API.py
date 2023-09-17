import requests
import json
import os
import numpy
from dotenv import load_dotenv
class Packing:

    req = requests
    #.envはトークンを格納しているファイル
    load_dotenv(".env")
    token = os.getenv("TOKEN")
    match_url = os.getenv("MATCH_URL")
    map_url = []
    header = {"procon-token":token}
    query = {"token":token}

    def get_match(self):
        """
        試合一覧取得API
        """
        match = self.req.get(self.match_url, self.token)

        for num in range(len(match.matches)):
            self.map_url.append(self.match_url + str(match[num]))
    
    def get_map(self,match_num):
        """
        試合状態取得API
        """
        map = self.req.get()
        return (map.board,)

    def post_act(self,next_turn,acts):
        self.req.post(self.url,data = self.data,header = self.header,turn = next_turn,actions = acts)

        