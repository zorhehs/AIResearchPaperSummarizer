import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

if __name__ == "__main__":
    for model in client.models.list():
        print(model.name, "-", model.supported_actions)
