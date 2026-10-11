import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from app.config.settings import RETRIEVAL_MODEL_NAME, EMBEDDING_MAX_LENGTH


class Retriever:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(RETRIEVAL_MODEL_NAME)
        self.model = AutoModel.from_pretrained(RETRIEVAL_MODEL_NAME)  # TODO: Quantization
        self.model.eval()
        self.device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.max_length = EMBEDDING_MAX_LENGTH

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def tokenize(self, text: str) -> dict:
        return self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, truncation=True, max_length=self.max_length)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, truncation=True, max_length=self.max_length))

    def embed(self, text: str) -> np.ndarray:
        if not text:
            return np.array([])

        tokens = self.tokenize(text)
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = self.last_token_pool(outputs.last_hidden_state, tokens['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)  # L2

        return embeddings.detach().cpu().numpy().squeeze()

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        tokens = self.tokenize(texts)
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = self.last_token_pool(outputs.last_hidden_state, tokens['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.detach().cpu().numpy()


if __name__ == "__main__":
    retriever = Retriever()
    queries = ["How much protein should a female eat?", "Summit define"]
    docs = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
        "Definition of summit for English Language Learners. : 1 the highest point of a mountain : the top of a mountain."
    ]

    query_embeds = retriever.embed_batch(queries)
    doc_embeds = retriever.embed_batch(docs)

    scores = (query_embeds @ doc_embeds.T) * 100
    print(scores.tolist())