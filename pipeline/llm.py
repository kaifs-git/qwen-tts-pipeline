from typing import Generator
from groq import Groq

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Keep every response concise and conversational — 2-3 sentences max. "
    "No bullet points or markdown, plain natural speech only."
)


class GroqLLM:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, user_message: str) -> Generator[str, None, None]:
        self.history.append({"role": "user", "content": user_message})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            stream=True,
            max_tokens=150,
        )

        full_response = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_response += token
                yield token

        self.history.append({"role": "assistant", "content": full_response})

    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
