import os
from dotenv import load_dotenv

load_dotenv(".env")
token = os.getenv("TOKEN")
print(token)