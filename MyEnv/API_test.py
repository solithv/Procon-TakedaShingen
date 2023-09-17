from Field_API import API
from dotenv import load_dotenv

fa = API()
id_ = fa.get_match()
print(fa.get_field(id_))