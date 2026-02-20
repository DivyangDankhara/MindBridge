import os

from openai import OpenAI

from llm.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")

        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0.2,
        )
        return response.output_text.strip()
