import requests

url = "https://6a82-2400-2410-a121-7300-7542-357f-4380-8973.ngrok-free.app/api"

data = {"hoge": 1234}

response = requests.post(url, json=data)

print(response.text)