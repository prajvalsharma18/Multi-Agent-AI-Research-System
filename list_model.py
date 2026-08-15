"""List Gemini models available for your API key."""

from __future__ import annotations

import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

RECOMMENDED = (
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("GOOGLE_API_KEY not set in .env")

genai.configure(api_key=api_key)

print("Available models (generateContent):\n")
models = []
for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        short_name = model.name.removeprefix("models/")
        models.append(short_name)
        tag = "  <- recommended" if short_name in RECOMMENDED else ""
        print(f"  {short_name}{tag}")

print(f"\nTotal: {len(models)}")
print("\nSuggested GEMINI_MODEL for this project:")
print("  GEMINI_MODEL=gemini-3.5-flash-lite")
