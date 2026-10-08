"""List models available to the configured OpenAI account."""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

for model in sorted(client.models.list().data, key=lambda item: item.id):
    print(model.id)
