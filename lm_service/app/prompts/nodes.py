from langchain_core.prompts import ChatPromptTemplate


SYSTEM_MSG = """You analyze text for the purpose of building a knowledge graph.
Your task is to extract as many relevant nodes as possible (high recall), that can function as nodes in the knowledge graph.

Specifically identify:
- persons (individual human actors),
- organizations (companies, institutions, movements),
- locations (geographical and administrative places),
- events (historical, social, political, technical, scientific, etc.),
- time expressions (dates, periods, time ranges),
- key thematic concepts (ideas, technologies, phenomena, theories),
- indicators and metrics (economic, financial, scientific, statistical, e.g., GDP growth, inflation, unemployment rate).

**Rules:**
- Return only semantically meaningful elements, that can exist independently as nodes in a knowledge graph.
- Use canonical names where possible (e.g. “European Union” instead of abbreviations, if context allows). Avoid returning multiple surface forms of the same entity or concept. If different names refer to the same real-world entity (e.g. „ChRL”, „Chiny”, „Chińska Republika Ludowa”), return only one canonical form, chosen based on common usage in context.
- Numeric values that are part of an indicator (like percentages, monetary amounts, growth rates) must be kept together if the context clearly indicates they belong to a single measure (e.g., "17,9%", "1,2 mln USD", "stopa inflacji 3,5%").
- Do not duplicate items.
- Do not add category labels, keys, explanations or metadata.
- Return everything in Polish.

**Output format:**
- A single line of text.
- All extracted elements separated by SEMICOLONS (;).
- No JSON, no lists, no bullet points, no headings, no comments."""

USER_EX_1 = """Text:
Donald Trump nie ogranicza się do przebudowy Białego Domu. Zmienia też to, jak Amerykanie patrzą
na świat. Wenezuela liczy się dla niego bardziej niż Ukraina, bo to z Ameryki Łacińskiej
pochodzą największe kłopoty dzisiejszych Stanów: nielegalna imigracja i przemyt narkotyków.
Ważny jest również Pacyfik i rywalizacja z Chinami o hegemonię w tym regionie.
Trump uważa, że USA traciły na porządku, który powstał po zakończeniu zimnej wojny.
Wciąż trwa spór o to, czy przyjęcie ChRL do WTO było dobrą decyzją.
Jednocześnie Ameryka wiele zyskiwała, między innymi dzięki temu,
że dolar pozostawał walutą rezerwową świata."""

AI_EX_1 = """Donald Trump; Biały Dom; Stany Zjednoczone; Amerykanie; Wenezuela; Ukraina; Ameryka Łacińska; Nielegalna imigracja; Przemyt narkotyków; Pacyfik; Chiny; Hegemonia; Zimna wojna; Światowa Organizacja Handlu; Dolar; Waluta rezerwowa świata"""

USER_EX_2 = """Text:
Adam Nowakowski, główny analityk w banku Pekao.
Instytucja w najnowszym biuletynie podkreśla, że nakłady na odnawialne źródła energii w 2024 r. gwałtownie wzrosły
- w ujęciu rocznym o 25,4 proc. - co wynika z realizacji nowych projektów farm wiatrowych na Bałtyku."""

AI_EX_2 = """Adam Nowakowski; Bank Pekao; Nakłady na odnawialne źródła energii; 25,4%; 2024; Morze Bałtyckie; Farmy wiatrowe"""

USER_MSG = "Text:\n{text}"


NODES_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_EX_1),
    ("ai", AI_EX_1),
    ("user", USER_MSG)
])
