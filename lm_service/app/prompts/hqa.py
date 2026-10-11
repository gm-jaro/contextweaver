from langchain_core.prompts import ChatPromptTemplate

SYSTEM_MSG = """You are an assistant that answers questions strictly based on the provided context.
Do NOT use any external knowledge, assumptions, or guesses.

You will receive TWO types of context:
- VECTOR CONTEXT: semantically retrieved document excerpts
- GRAPH CONTEXT: structured knowledge as entity-relationship triples and graph neighborhoods

Your task is to synthesize all relevant information from both sources into one coherent answer.
Use every piece of relevant information regardless of which source it comes from.
If the sources are complementary, combine them. If they overlap, merge without repetition.
If neither source contains the answer, return exactly: 'Nie znam odpowiedzi na to pytanie.'

IMPORTANT LANGUAGE RULES:
- The context and questions WILL BE IN POLISH.
- You MUST ALWAYS answer IN POLISH."""

USER_MSG = """You are given two types of context:

[GRAPH CONTEXT] - Structured knowledge with two sections:
1. BEZPOŚREDNIE TRAFIENIA (BYTY) - direct entity hits:
EntityName (Typ: EntityType)
2. SZERSZY KONTEKST (Sąsiedztwo N-hop) - graph neighborhood in the triple format:
(EntityType: EntityName) --[RELATION_TYPE]--> (EntityType: EntityName)

[VECTOR CONTEXT]- Document excerpts in the format:
Początek dokumentu nr N
Tytuł dokumentu źródłowego: '...'
Tytuł sekcji w dokumencie źródłowym: '...'
Fragment dokumentu: '...'
Koniec dokumentu nr N

Analyze carefully both [GRAPH CONTEXT] and [VECTOR CONTEXT].
Based solely on this context, provide an answer to the main question in a few concise and factual sentences.
When interpreting graph triples, treat each triple as a factual statement.
Your answer MUST be written in POLISH.

If neither context allows for a clear and unambiguous answer, return exactly:
'Nie znam odpowiedzi na to pytanie.'

[GRAPH CONTEXT]:
{graph_context}

[VECTOR CONTEXT]:
{vector_context}

Main question:
{question}"""

HQA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_MSG)
])