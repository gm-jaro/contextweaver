import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _load_openai_key() -> str:
    """OpenAI key: first a file holding just the key (OPENAI_KEY_FILE), then the environment variable."""
    key_file = Path(os.environ.get("OPENAI_KEY_FILE", ""))
    try:
        if key_file.is_file():
            raw = key_file.read_text(encoding="utf-8-sig").strip()
            raw = "".join(ch for ch in raw if ch.isprintable() and not ch.isspace())
            if "=" in raw:
                raw = raw.split("=", 1)[1].strip().strip('"').strip("'")
            if raw:
                os.environ["OPENAI_API_KEY"] = raw
                return raw
    except OSError:
        pass
    return os.environ.get("OPENAI_API_KEY", "").strip()


OPENAI_API_KEY = _load_openai_key()

QDRANT_URL = "http://localhost:6334"
MODEL_CLIENT_URL = "http://localhost:8080"
REDIS_URL="redis://localhost:6379"
MONGO_URL = f"mongodb://admin:{os.environ['MONGO_PASSWORD']}@localhost:27017"
PROJECT_VERSION="0.1"
PORT=8060
LOG_LEVEL="DEBUG"
NEO4J_URL="neo4j://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
MAX_DOCS=5

if OPENAI_API_KEY:
    LLM_MODEL_NAME="gpt-5-nano"
else:
    LLM_MODEL_NAME="gpt-oss:20b" # "SpeakLeash/bielik-11b-v3.0-instruct:Q8_0"  # "gpt-oss:20b" # "jaro-jm/llm:latest"

# TODO: Experiments with choosing the agent llm: gpt-oss:20b, magistral:24b, devstral-small-2:24b,
#  glm-4.7-flash, qwen3-vl:30b, nemotron-3-nano:latest, deepseek-r1:14b, cogito:32b

NUM_OF_AGENT_ITERATIONS=25
LLM_AGENT_NUM_CTX=131000
LLM_AGENT_REASONING="medium"
MAX_RETRIES = 2
