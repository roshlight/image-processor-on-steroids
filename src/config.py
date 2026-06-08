from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / 'prompts'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'

load_dotenv(override=True)

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')

ALIBABA_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

QWEN_MODEL = "qwen3.5-plus"

# Чтобы в случае чего сразу получить ошибку на месте, а не когда мы сделаем запрос по апи. 
# Предотвращаем ситуацию, когда падает непонятно почему
def get_dashscope_api_key() -> str:
    if not DASHSCOPE_API_KEY:
        raise RuntimeError('Добавь в .env DASHSCOPE_API_KEY')
    return DASHSCOPE_API_KEY


# Аналогично, безопасно читаем промпт файл
def read_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Файл с промптом не найден: {path}")
    return path.read_text(encoding='utf-8')
