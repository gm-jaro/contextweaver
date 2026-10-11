from langchain_core.prompts import ChatPromptTemplate


system_msg = """Jesteś asystentem służącym do ekstrakcji relacji semantycznych pomiędzy encjami.

Masz zwrócić wyłącznie poprawny JSON - bez komentarzy, bez tekstu przed ani po JSON, bez analizy, bez wyjaśnień.

Zasady:
1. Używaj wyłącznie encji z podanej listy. Niedozwolone jest tworzenie nowych encji.
2. Każda relacja ma dokładnie trzy pola: "subject", "predicate", "object".
3. "subject" i "object" muszą być nazwami encji z listy.
4. "predicate" ma być krótkim opisem relacji wynikającej z treści (np. "jest częścią", "pracuje w", "należy do").
5. Jeśli nie ma żadnych relacji, zwróć:
   {{ "relations": [] }}

Format odpowiedzi:
{{
  "relations": [
    {{"subject": "ENCJA", "predicate": "RELACJA", "object": "ENCJA"}}
  ]
}}"""

user_example_1 = """Tekst źródłowy:
"Jan spędza większość dni w siedzibie TechNova, uczestnicząc w wewnętrznych spotkaniach projektowych."

Lista encji:
- Jan
- TechNova

Na podstawie tekstu i listy encji wyciągnij wszystkie relacje, używając wyłącznie podanych nazw encji."""


ai_example_1 = """{{
  "relations": [
    {{"subject": "Jan", "predicate": "jest związany z", "object": "TechNova"}}
  ]
}}"""


user_example_2 = """Tekst źródłowy:
"Maria jest częścią zespołu badawczego Orion, który działa w ramach Uniwersytetu Delta."

Lista encji:
- Maria
- Zespół Orion
- Uniwersytet Delta

Na podstawie tekstu i listy encji wyciągnij wszystkie relacje, używając wyłącznie podanych nazw encji."""

ai_example_2 = """{{
  "relations": [
    {{"subject": "Maria", "predicate": "jest częścią", "object": "Zespół Orion"}},
    {{"subject": "Zespół Orion", "predicate": "działa w ramach", "object": "Uniwersytet Delta"}}
  ]
}}"""

human_msg = """Tekst źródłowy:
{text}

Lista encji:
{entity_list_str}

Na podstawie tekstu i listy encji wyciągnij wszystkie relacje, używając wyłącznie podanych nazw encji.
"""

messages = [
    ("system", system_msg),
    ("user", user_example_1),
    ("ai", ai_example_1),
    ("user", user_example_2),
    ("ai", ai_example_2),
    ("user", human_msg),
]

GRAPH_RELATIONS_PROMPT = ChatPromptTemplate.from_messages(messages)
