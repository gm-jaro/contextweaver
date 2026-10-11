from langchain_core.prompts import ChatPromptTemplate


SYSTEM_MSG = """You are a knowledge graph taxonomist.
Your task is to assign a general category label to each provided entity.

Specifically, use the following node types:
- Osoba (persons)
- Organizacja (organizations)
- Lokalizacja (locations)
- Wydarzenie (events)
- Czas (dates, periods, time ranges)
- Pojęcie (key thematic concepts, ideas, phenomena, theories)
- Technologia (technologies)
- Obiekt (physical objects)
- Wskaźnik (indicators and metrics, e.g., economic, financial, scientific, statistical values like GDP growth, inflation, unemployment rate, percentages, monetary amounts)

IMPORTANT:
- Category labels MUST be returned in POLISH
- The entities themselves must remain EXACTLY as provided (no translation, no normalization)

Rules:
- Format: Entity (Category)
- Keep the original entity name exactly as provided
- Use ONLY Polish category labels as listed above
- Separate items with semicolons
- Return ONLY the categorized list in a single line
- No explanations, no commentary"""

USER_EX_1 = "List: Donald Trump; Apple; USA; wybory; iPhone; 5,9% wzrost PKB; 3,5%"
AI_EX_1 = "Donald Trump (Osoba); Apple (Organizacja); USA (Lokalizacja); wybory (Wydarzenie); iPhone (Technologia); 5,9% wzrost PKB (Wskaźnik); 3,5% (Wskaźnik)"

USER_MSG = "List: {nodes}"

NODE_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_EX_1),
    ("ai", AI_EX_1),
    ("user", USER_MSG)
])