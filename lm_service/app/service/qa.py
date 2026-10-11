import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from app.clients.vectorizer import VectorizerClient
from app.config import settings
from app.models.qa_dto import QAResponse, QARAGMode
from app.prompts import SUBQ_SECTION
from app.prompts.gqa import GQA_PROMPT
from app.prompts.qa import QA_PROMPT
from app.prompts.hqa import HQA_PROMPT
from app.prompts.qgen import QGEN_PROMPT
from loguru import logger


def _with_subq(prompt: ChatPromptTemplate) -> ChatPromptTemplate:
    *head, last = prompt.messages
    return ChatPromptTemplate.from_messages([*head, ("user", last.prompt.template + SUBQ_SECTION)])

def _log_reasoning(reasoning: str, question: str) -> None:
    if not reasoning:
        return
    border = "-" * 60
    logger.debug(
        f"\n{border}\n"
        f"  REASONING  ->  {question[:80]}\n"
        f"{border}\n"
        f"{reasoning}\n"
        f"{border}"
    )


class QAService:
    def __init__(self, vectorizer_client: VectorizerClient):
        self.vectorizer_client = vectorizer_client

        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip():
            self.llm = ChatOpenAI(model=settings.LLM_MODEL_NAME, temperature=0.0, api_key=settings.OPENAI_API_KEY,
                                  reasoning = {"effort": settings.REASONING, "summary": "auto"})
            logger.info(f"Using OpenAI model: {settings.LLM_MODEL_NAME}")
        else:
            self.llm = ChatOllama(model=settings.LLM_MODEL_NAME, validate_model_on_init=True,
                                  reasoning=settings.REASONING, temperature=0.0,
                                  num_ctx=131000, num_predict=-2)
            logger.info(f"Using Ollama model: {settings.LLM_MODEL_NAME}")

        base_prompts = {QARAGMode.VANILLA: QA_PROMPT,
                             QARAGMode.GRAPH_CHUNK: QA_PROMPT,
                             QARAGMode.GRAPH: GQA_PROMPT,
                             QARAGMode.HYBRID: HQA_PROMPT}

        self.chains = {mode: prompt | self.llm for mode, prompt in base_prompts.items()}
        self.chains_with_subq = {mode: _with_subq(prompt) | self.llm for mode, prompt in base_prompts.items()}

    async def answer_question(self, question: str, rag_mode: QARAGMode, gen_subquestions: bool) -> QAResponse:
        queries = [question]
        invoke_kwargs = {"question": question}
        subquestions = None
        if gen_subquestions:
            subquestions = await self.gen_subquestions(question)
            queries += subquestions
            invoke_kwargs["subquestions"] = "\n".join(subquestions)

        chains = self.chains_with_subq if gen_subquestions else self.chains

        if rag_mode == QARAGMode.HYBRID:
            vanilla_ctx, graph_ctx = await asyncio.gather(self.vectorizer_client.get_context(queries, QARAGMode.VANILLA),
                                                          self.vectorizer_client.get_context(queries, QARAGMode.GRAPH))

            response = await chains[QARAGMode.HYBRID].ainvoke({"vector_context": vanilla_ctx,
                                                               "graph_context": graph_ctx,
                                                               **invoke_kwargs})

            source_context = f"[GRAPH]\n{graph_ctx}\n\n\n[VECTOR]\n{vanilla_ctx}"

        else:
            source_context = await self.vectorizer_client.get_context(queries, rag_mode)
            response = await chains[rag_mode].ainvoke({"context": source_context, **invoke_kwargs})

        reasoning = "\n".join(
            (getattr(b, "reasoning", None) or b.get("reasoning", ""))
            for b in response.content_blocks
            if (getattr(b, "type", None) or b.get("type")) == "reasoning"
        ).strip()

        _log_reasoning(reasoning, question)

        answer = getattr(response, "text", None) or str(response.content)

        return QAResponse(answer=answer,
                          source_context=source_context,
                          reasoning=reasoning,
                          subquestions=subquestions)

    async def gen_subquestions(self, question: str) -> list[str]:
        chain = QGEN_PROMPT | self.llm
        response = await chain.ainvoke({"text": question})
        raw_content = response.content
        questions = [q.strip() for q in raw_content.splitlines() if q.strip()]
        return questions
