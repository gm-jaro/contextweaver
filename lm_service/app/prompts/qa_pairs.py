from langchain_core.prompts import ChatPromptTemplate


SYSTEM_MSG = """You are an expert assistant for generating high-quality question–answer pairs from text using provided entities.

Rules:
- Generate 6-8 question–answer pairs in Polish.
- Use ONLY entities listed below to form questions:\n{nodes}
- Do NOT ask generic or meta-level questions about the text itself. Forbidden example of a question: "Jakie jest główne zagadnienie omawiane w tekście?"
- Questions must test understanding of the content, not summarize it at a high level.
- Avoid trivial, vague, or purely definitional questions. Questions and answers MUST cover ONLY the most important aspects of the text.
- Answers must be concise, precise, and based only on the provided text.
- Avoid pronouns like "tych", "tamte", "on", "jego" (these, those, he, his) if they refer to entities mentioned in previous sentences. Replace them with full entity names.
- Resolve ambiguities by clearly attributing statements, claims, or actions to specific people, entities or organizations, indicating who thinks, says, or does what, rather than leaving information vague or general.
- Use canonical names where possible (e.g. “European Union” instead of abbreviations, if context allows).
Avoid returning multiple surface forms of the same entity or concept.
If different names refer to the same real-world entity (e.g. „ChRL”, „Chiny”, „Chińska Republika Ludowa”), return only one canonical form, chosen based on common usage in context.

Output format:
- Return ONLY valid JSON.
- The JSON must be a list of objects.
- Each object must have exactly two fields: "question" and "answer".
- Do NOT include markdown, comments, or any text outside the JSON.

Example output:
[
  {{
    "question": "Jaką rolę pełni mechanizm X w opisywanym procesie?",
    "answer": "Mechanizm X odpowiada za ..."
  }},
  {{
    "question": "Jakie konsekwencje wynikają z zastosowania rozwiązania Y?",
    "answer": "Zastosowanie rozwiązania Y prowadzi do ..."
  }}
]"""

USER_MSG = """Generate question–answer pairs from the following text:\n{text}"""

QA_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_MSG)
])
