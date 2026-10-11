from langchain_core.prompts import ChatPromptTemplate

# v7: one call instead of three (node extraction with the chunk prompt + classifier + relation labels).
# Result: canonical entities with periphrases resolved, aliases for full-text search, keywords,
# the expected answer type and the question rewritten without periphrases.

SYSTEM_MSG = """You analyze a user question written in Polish in order to retrieve facts from a knowledge graph
built from Polish press articles (economy, politics, business). Return ONLY one JSON object, no prose, no code fences.

JSON fields:
- "entities": list of objects, each with:
    "name": canonical Polish name in nominative case, exactly as a press article would write it,
    "type": one of ["Osoba", "Organizacja", "Lokalizacja", "Wydarzenie", "Czas", "Pojęcie", "Technologia", "Obiekt", "Wskaźnik"],
    "aliases": other surface forms likely to appear in articles (abbreviations, full names, common variants), 0-4 items.
- "keywords": 3-6 short Polish noun phrases describing WHAT the question asks about (phenomena, indicators, actions),
  not repeating the entities.
- "answer_type": one of ["osoba", "organizacja", "liczba", "data", "przyczyna", "skutek", "opis", "lista", "porównanie", "tak/nie"].
- "canonical_question": the question rewritten in plain, explicit Polish, with every descriptive reference replaced
  by the canonical name.

Rules:
- Resolve descriptive references (peryfrazy) to the concrete named entity whenever it is unambiguous for a reader of
  Polish press: "kraj nad Wisłą" -> "Polska", "duży krajowy koncern paliwowy" -> "Orlen", "wspólnota gospodarcza"
  -> "Unia Europejska", "prezes banku centralnego" -> "Adam Glapiński", "wschodni członek Unii" -> "Polska".
  If genuinely ambiguous, keep a short generic phrase as the name (type "Pojęcie").
- Include dates, years and named periods as entities of type "Czas"; named indicators (PKB, inflacja, WIG20) as "Wskaźnik".
- Include persons even when only a surname is given ("Łukaszenka" -> name "Alaksandr Łukaszenka", aliases ["Łukaszenka", "Łukaszenko"]).
- Keep names short (max 5 words). Do not invent entities that are not implied by the question.
- Everything in Polish."""

USER_EX_1 = "Question: Czym różniło się podejście dużego krajowego koncernu paliwowego do kształtowania cen detalicznych od podejścia władz państwowych do limitowania wywozu towarów za granicę?"
AI_EX_1 = """{{"entities": [{{"name": "Orlen", "type": "Organizacja", "aliases": ["PKN Orlen", "koncern paliwowy"]}}, {{"name": "Polska", "type": "Lokalizacja", "aliases": ["rząd", "władze państwowe"]}}, {{"name": "Ceny paliw", "type": "Wskaźnik", "aliases": ["ceny detaliczne paliw", "cena Pb95"]}}], "keywords": ["polityka cenowa", "ceny detaliczne", "ograniczenie eksportu", "wywóz towarów", "łagodzenie szoku cenowego"], "answer_type": "porównanie", "canonical_question": "Czym różniło się podejście Orlenu do kształtowania cen detalicznych paliw od podejścia polskiego rządu do ograniczania eksportu towarów?"}}"""

USER_EX_2 = "Question: Co analitycy DM BOŚ mówią o Inter Cars?"
AI_EX_2 = """{{"entities": [{{"name": "DM BOŚ", "type": "Organizacja", "aliases": ["Dom Maklerski BOŚ", "BOŚ"]}}, {{"name": "Inter Cars", "type": "Organizacja", "aliases": ["akcje Inter Cars"]}}], "keywords": ["rekomendacja analityków", "cena docelowa akcji", "raport analityczny"], "answer_type": "opis", "canonical_question": "Co analitycy DM BOŚ mówią w swojej rekomendacji o spółce Inter Cars?"}}"""

USER_MSG = "Question: {text}"

QUESTION_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_EX_1),
    ("ai", AI_EX_1),
    ("user", USER_EX_2),
    ("ai", AI_EX_2),
    ("user", USER_MSG)
])
