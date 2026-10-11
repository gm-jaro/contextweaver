from typing import List
import torch
from langchain_core.runnables import RunnableSerializable
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from app.config.settings import NER_MODELS
from loguru import logger
import os
from langchain_ollama import ChatOllama
from app.config import settings
from experiments.graph_relations import GRAPH_RELATIONS_PROMPT


class NER:
    def __init__(self, lang: str):
        if lang not in NER_MODELS:
            raise ValueError(f"No model for language: {lang}")

        self.lang = lang

        logger.debug(f"Using language: {self.lang}")
        logger.debug(f"HF_HOME: {os.getenv('HF_HOME')}")

        self.model_name = NER_MODELS[lang]

        self.device = 0 if torch.cuda.is_available() else -1
        logger.debug(f"Device detected: {torch.cuda.is_available()}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)

        self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer,
                                 grouped_entities=True, device=self.device)

    def deduplicate(self):
        """many identical entities e.g. synonyms, abbreviations merged into one entity, resolve_coreference"""
        ...

    def link(self):
        """validation, linking with wiki"""
        ...

    def __call__(self, text: str):
        return self.pipeline(text)


class RelationsPipeline:
    def __init__(self, ner: NER):
        self.ner = ner
        self.llm_chain = GRAPH_RELATIONS_PROMPT | ChatOllama(model=settings.LLM_MODEL_NAME, validate_model_on_init=True, reasoning=False, temperature=0.0)  # keep_alive

    def extract_relations(self, text: str, entities: List[str]):
        if not self.llm_chain:
            raise RuntimeError("LLM chain is not initialized.")

        entity_list_str = "\n".join(f"- {e}" for e in entities)

        logger.debug(f"Entity list: {entity_list_str}, \n text: {text}")

        resp = self.llm_chain.invoke({
            "text": text,
            "entity_list_str": entity_list_str
        }).content

        logger.debug(f"resp: {resp}")

        if isinstance(resp, str):
            try:
                import json
                return json.loads(resp)
            except Exception:
                return {"relations": []}

        elif isinstance(resp, dict):
            return resp

        else:
            try:
                import json
                return json.loads(str(resp))

            except Exception:
                return {"relations": []}

    def __call__(self, text: str, entities: List[str]):
        return self.extract_relations(text, entities)


if __name__ == "__main__":
    from pprint import pprint

    demo_text = """W dniu 14 marca 2024 roku, profesor Adam Szymański, 
    wybitny językoznawca z Uniwersytetu Jagiellońskiego w Krakowie, 
    otrzymał Nagrodę Nobla. W swoim przemówieniu cytował: 
    "The only true wisdom is in knowing you know nothing," (Sokrates). 
    Następnie ogłosił, że przekaże 50% nagrody, czyli około 4.5 mln PLN, 
    fundacji Wielka Orkiestra Świątecznej Pomocy (WOŚP).
    Spotkanie odbyło się w warszawskim Pałacu Kultury i Nauki, a relację 
    przeprowadziła telewizja TVN."""

    ner = NER("pl")
    entities = [e['word'] for e in ner(demo_text)]
    pprint(entities)

    rel = RelationsPipeline(ner)
    pprint(rel(demo_text, entities))