from app.config.settings import RERANKING_MODEL_NAME, EMBEDDING_MAX_LENGTH
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


class Reranker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKING_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            RERANKING_MODEL_NAME,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            device_map=None
        )
        self.model.eval()
        self.device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.max_length = EMBEDDING_MAX_LENGTH
        
    
    def score_batch(self, query: str, passages: list[str]) -> list[float]:
        if not query or not passages:
            return []

        texts = [f"{query}</s></s>{p}" for p in passages]

        tokens = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**tokens)

        scores = output.logits.detach().cpu().float().numpy()
        scores = np.squeeze(scores)
        return scores.tolist() if scores.ndim > 0 else [float(scores)]

    
if __name__ == "__main__":
    reranker = Reranker()
    query = "Jak dożyć 100 lat?"
    candidates = [
        "Trzeba zdrowo się odżywiać i uprawiać sport.",
        "Trzeba pić alkohol, imprezować i jeździć szybkimi autami.",
        "Gdy trwała kampania politycy zapewniali, że rozprawią się z zakazem niedzielnego handlu."
    ]
    results = reranker.score_batch(query, candidates)
    for score, candidate in zip(results, candidates):
        print(f"{score:.4f} - {candidate}")