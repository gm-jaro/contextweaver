from langchain_core.prompts import ChatPromptTemplate

SYSTEM_MSG = """You are an expert in entity resolution for knowledge graphs.
Your task is to determine whether newly extracted entities refer to the same real-world objects as entities already existing in a knowledge graph.

RULES:
- Match ONLY when you are confident both names refer to the exact same real-world entity.
- Name specificity differences ARE acceptable matches: "Putin" = "Władimir Putin", "USA" = "Stany Zjednoczone".
- Same name but different context/type MUST NOT match: "Jan Kowalski (Autor)" doesnt match "Jan Kowalski (Polityk)".
- Abbreviations and aliases are acceptable: "UE" = "Unia Europejska".
- If no candidate is a confident match, return null for matched_id.

OUTPUT FORMAT:
Return ONLY a valid JSON array. No explanation, no markdown, no code fences.
Each element must have exactly these fields:
- "input_name": string (the original name from INPUT ENTITIES)
- "matched_id": string or null (node's id of the matching candidate, or null)
- "matched_name": string or null (name of the matched candidate, or null)
- "confidence": integer between 0-10 
- "reasoning": string (one sentence justification)
"""

USER_MSG = """CURRENT CHUNK (source context for the NEW entities being integrated):
{chunk_text}

---
{source_section}

---
INPUT ENTITIES (newly extracted, to be resolved):
{input_entities}

---
CANDIDATES FROM GRAPH (existing nodes, with all source chunks they appeared in):
{candidates}

---
For each INPUT ENTITY, decide if it matches one of its candidates or is a new entity.
Return a JSON array of resolutions."""

ND_V4_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_MSG)
])