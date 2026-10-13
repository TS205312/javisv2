"""
Cấu hình chung cho Autonomous Agent.
Đọc từ .env và biến môi trường hệ thống.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Fast/light strategy for the Ryzen 5 4600G: cloud first, 1.5B Ollama offline backup.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
KILO_API_KEY = os.getenv("KILO_API_KEY", "")
KILO_MODEL = os.getenv("KILO_MODEL", "kilo-auto/free")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b")
OLLAMA_NUM_PARALLEL = int(os.getenv("OLLAMA_NUM_PARALLEL", "1"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_PRIMARY_MODEL = os.getenv(
    "OPENROUTER_PRIMARY_MODEL",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
)
OPENROUTER_FALLBACK_MODEL = os.getenv(
    "OPENROUTER_FALLBACK_MODEL",
    "nvidia/nemotron-3.5-lightning:free",
)
_DEFAULT_OPENROUTER_FREE_POOL = (
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "poolside/laguna-s-2.1:free",
    "cohere/north-mini-code:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "liquid/lfm-2.5-2.6b:free",
    "thinkingmachines/inkling:free",
)
_openrouter_pool_setting = os.getenv("OPENROUTER_FREE_POOL")
OPENROUTER_FREE_POOL = (
    tuple(
        model.strip()
        for model in _openrouter_pool_setting.split(",")
        if model.strip()
    )
    if _openrouter_pool_setting is not None
    else _DEFAULT_OPENROUTER_FREE_POOL
)
_legacy_openrouter_model = os.getenv(
    "OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"
)
OPENROUTER_MODEL = (
    OPENROUTER_FREE_POOL[0] if OPENROUTER_FREE_POOL else _legacy_openrouter_model
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
BLENDER_PATH = os.getenv("BLENDER_PATH", "blender")
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

# Timeout mặc định cho mỗi agent trong swarm (giây)
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "300"))

# Số agent mặc định trong swarm
DEFAULT_AGENT_COUNT = int(os.getenv("DEFAULT_AGENT_COUNT", "4"))

def get_llm_api_key():
    return GEMINI_API_KEY
