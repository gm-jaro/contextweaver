from langchain_core.prompts import ChatPromptTemplate


SYSTEM_MSG = """You are an assistant that answers questions strictly based on the provided context.
Do not add any external knowledge, assumptions, or interpretations.

LANGUAGE REQUIREMENTS:
- The context and questions WILL BE IN POLISH.
- You MUST ALWAYS respond IN POLISH.

If the answer is not explicitly present in the context, return exactly:
'Nie znam odpowiedzi na to pytanie.'.

Do not add any information outside of the given context.
If you do not know the answer or are uncertain, return exactly:
'Nie znam odpowiedzi na to pytanie.'."""


USER_MSG = """You are given context consisting of one or more document excerpts.
Each excerpt follows this format:
Początek dokumentu nr N
Tytuł dokumentu źródłowego: '...'
Tytuł sekcji w dokumencie źródłowym: '...'
Fragment dokumentu: '...'
Koniec dokumentu nr N

Analyze carefully all of the provided excerpts.
Based solely on this context, provide an answer to the main question in a few concise and factual sentences.
Your answer MUST be written IN POLISH.

If the context does not allow for a clear and unambiguous answer to the main question, return exactly:
'Nie znam odpowiedzi na to pytanie.'.

Context:
{context}

Main question:
{question}"""


QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_MSG)
])