from langchain_ollama import ChatOllama
from app.prompts.summary import SUMMARY_GENERATOR_PROMPT
from loguru import logger
from app.config.settings import LLM_MODEL_NAME


def generate_summary(text: str):
    logger.info("Generating summary using model=%s", LLM_MODEL_NAME)
    logger.debug("Input text length: %d characters", len(text))
    llm_chain = SUMMARY_GENERATOR_PROMPT | ChatOllama(model=LLM_MODEL_NAME, temperature=0, reasoning=False, num_ctx=10000, num_predict=25000)

    try:
        summary = llm_chain.invoke({"text": text}).content
        logger.info("Summary generated successfully")
        logger.debug("Summary length: %d characters", len(summary))
        return summary

    except Exception:
        logger.exception("Failed to generate summary")
        raise
