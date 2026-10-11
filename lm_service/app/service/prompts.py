import yaml
from app.config.settings import PROMPTS_PATH


_prompts_cache = None

async def load_prompts() -> dict:
    global _prompts_cache
    if _prompts_cache is None:
        if not PROMPTS_PATH.exists():
            raise FileNotFoundError(f"Prompts file not found: {PROMPTS_PATH}")
        with PROMPTS_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            _prompts_cache = {k: str(v).strip() for k, v in data.items() if v}
    return _prompts_cache.copy()
