
from dotenv import load_dotenv
load_dotenv()

import os
from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(api_version="v1"),
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="""
    INSTRUCTIONS:
    You are a helpful meeting assistant.
    Summarize the given sentence clearly and concisely.

    TASK:
    The team will deploy the application on Friday.
    """,
    config=types.GenerateContentConfig(
        temperature=0.1,
    ),
)

print(response.text)
