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

LOG_LEVEL = "DEBUG"
REASONING = "medium"
PROJECT_NAME = "LLM Service"
PROJECT_VERSION = "0.0.1"
RERANKING_MODEL_NAME = "sdadas/polish-reranker-roberta-v3"
VLM_MODEL_NAME = "glm-ocr:latest"  # "jaro-jm/vlm:latest"  # gemma3:12b

if OPENAI_API_KEY:
    LLM_MODEL_NAME = "gpt-5-nano"
else:
    LLM_MODEL_NAME = "glm-4.7-flash:q4_K_M" # "SpeakLeash/bielik-11b-v3.0-instruct:Q8_0"  # "gpt-oss:20b" # "jaro-jm/llm:latest"

RETRIEVAL_MODEL_NAME = "sdadas/stella-pl-retrieval-8k"  # "BAAI/bge-multilingual-gemma2"  # granite-embedding:278m  # embeddinggemma-300  # shuyuej/bge-multilingual-gemma2-GPTQ  # BAAI/bge-m3
EMBEDDING_MAX_LENGTH = 2048  # For reranker and retriever
PORT=8080

NER_MODELS = {
    "pl": "pczarnik/herbert-base-ner",
    "en_wiki": "Babelscape/wikineural-multilingual-ner",
    "en": "dslim/bert-base-NER"
}

GLM_ONFIG_PATH = str(Path(__file__).parent / "vlm_config.yml")

MODEL_FILES = {
    # VLM_MODEL_NAME: "/opt/models/InternVL3_5-GPT-OSS-20B-A4B-Preview",
    # VLL_MODEL_NAME: "models/gemma3_12b_it_q8_0.gguf",
    # RETRIEVAL_MODEL_NAME: ...
    # RERANKING_MODEL_NAME: ...
    # LL_MODEL_NAME: ...
    # NER_MODEL_NAME: ...
}

PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "prompts.yaml"

OLLAMA_API_URL = "http://localhost:11434/"
VECTORIZER_URL = "http://localhost:8060"

NUM_SIM_QUESTIONS_TO_GENERATE = "1-2"
QUESTION_NUM_OF_RELATIONS = "10"
