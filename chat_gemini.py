from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# client = genai.Client(api_key='AIzaSyCUjLTu_PUSoLhUFoB2g9SHUOdzh3PBklI')

 
client = genai.Client()

response = client.models.generate_content(
    model='gemini-2.0-flash-001', contents='why is sky is blue?'
)
print(response.text)