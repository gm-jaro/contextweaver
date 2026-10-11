from langchain_core.prompts import ChatPromptTemplate
from app.config.settings import QUESTION_NUM_OF_RELATIONS


SYSTEM_MSG = f"""Analyze the user questions and extract ONLY the semantic relationship labels (predicates).
Your task is to identify how entities are connected or the intent of the queries.

Rules:
- Return ONLY the relationship labels.
- Do NOT return nodes, subjects, or objects.
- Maximum {QUESTION_NUM_OF_RELATIONS} labels.
- Use Polish language.
- Each label in a new line.
- Relationships should be concise (maximum of 3 words).
- No numbering, no JSON, no headers, no fluff."""

USER_EX_1 = "Questions: Jaką rolę pełni Władimir Putin w Rosji?\nJak inflacja wpływa na ceny złota?"
AI_EX_1 = """PEŁNI_ROLĘ
JEST_LIDEREM
ZAJMUJE_STANOWISKO
WPŁYWA
ODDZIAŁUJE
KORELUJE_Z"""

USER_EX_2 = "Questions: Text: Dlaczego palenie papierosów powoduje raka płuc?\n Kto założył Apple, gdzie ma siedzibę i co produkuje?\nCzy Bitcoin jest lepszy od Ethereum pod kątem szybkości?"
AI_EX_2 = """POWODUJE
WYWOŁUJE_CHOROBĘ
ZWIĄZEK_PRZYCZYNOWY
ZAŁOŻYCIEL
LOKALIZACJA_SIEDZIBY
LOKALIZACJA
PRODUKUJE
PORÓWNANIE
WYDAJNOŚĆ_TECHNICZNA
POSIADA_CECHĘ"""

USER_EX_3 = "Questions: Kiedy wybuchła II Wojna Światowa i jak długo trwała?\nDo kogo należy InPost i w jakich krajach działa?"
AI_EX_3 = """DATA_ROZPOCZĘCIA
CZAS_TRWANIA
WYDARZENIE_HISTORYCZNE
WŁAŚCICIEL
STRUKTURA_WŁASNOŚCIOWA
ZASIĘG_DZIAŁANIA"""

USER_MSG = "Questions: {text}"

QUESTION_RELATIONS_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_EX_1),
    ("ai", AI_EX_1),
    ("user", USER_EX_2),
    ("ai", AI_EX_2),
    ("user", USER_EX_3),
    ("ai", AI_EX_3),
    ("user", USER_MSG)
])