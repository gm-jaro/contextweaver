import os
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = "DEBUG"
PROJECT_NAME = "Ingestor Service"
OCR_CONFIDENCE_THRESHOLD = 0.85  # Minimum confidence to accept OCR result without correction
PROJECT_VERSION = "0.0.1"
PADDLE_DIR = os.getenv("PADDLE_DIR", "models/Paddle")
MODEL_CLIENT_URL = "http://localhost:8080"
OLLAMA_URL = "http://localhost:11434"


DO_OCR = False
DO_ENRICHMENT = False

PORT=8070
HOST="0.0.0.0"

# ALLOWED_FORMATS = [
#             InputFormat.PDF,
#             InputFormat.IMAGE,
#             InputFormat.DOCX,
#             InputFormat.HTML,
#             InputFormat.PPTX,
#             InputFormat.CSV,
#             InputFormat.MD,
#         ]
REDIS_URL = "redis://localhost:6379"
MONGO_URL = f"mongodb://admin:{os.environ['MONGO_PASSWORD']}@localhost:27017"
