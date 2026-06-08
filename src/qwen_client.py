from openai import OpenAI

from src.config import ALIBABA_BASE_URL, QWEN_MODEL, get_dashscope_api_key


def get_qwen_client() -> OpenAI:
    return OpenAI(
        api_key=get_dashscope_api_key(),
        base_url=ALIBABA_BASE_URL,
    )


def call_qwen_chat(messages, temperature: float = 0.0, max_tokens: int = 1200) -> str:
    client = get_qwen_client()

    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content
