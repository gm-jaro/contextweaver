import json
from typing import List, Dict, TypedDict
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langgraph.constants import END
from app.config import settings


class EntitiesOutput(BaseModel):
    persons: List[str] = []
    organizations: List[str] = []
    locations: List[str] = []

class EventsOutput(BaseModel):
    events: List[str] = []
    time_expressions: List[str] = []

class CoreConceptsOutput(BaseModel):
    core_concepts: List[str] = []

class QAEntry(BaseModel):
    concept: str
    qa: List[Dict[str, str]]

class QAGenerationOutput(BaseModel):
    entries: List[QAEntry]

class Triple(BaseModel):
    subject: str
    predicate: str
    object: str

class TriplesOutput(BaseModel):
    triples: List[Triple]


class GraphState(TypedDict):
    text: str
    entities: Dict[str, List[str]]
    events: Dict[str, List[str]]
    core_concepts: List[str]
    qa_pairs: List[Dict]
    triples: List[Dict]


llm_json = ChatOllama(model=settings.LLM_MODEL_NAME, temperature=0)
llm_text = ChatOllama(model=settings.LLM_MODEL_NAME, temperature=0)



ner_prompt = ChatPromptTemplate.from_messages([
    ("system", "Wydobądź osoby, organizacje oraz lokalizacje z tekstu. Zwróć JSON."),
    ("user", "Tekst:\n{text}\n\nZwróć JSON: {{\"persons\":[],\"organizations\":[],\"locations\":[]}}")
])

events_prompt = ChatPromptTemplate.from_messages([
    ("system", "Wydobądź wydarzenia i wzmianki czasowe. Zwróć JSON."),
    ("user", "Tekst:\n{text}\n\nZwróć JSON: {{\"events\":[],\"time_expressions\":[]}}")
])

concepts_prompt = ChatPromptTemplate.from_messages([
    ("system", "Wydziel główne pojęcia tematyczne. Zwróć JSON."),
    ("user", "Tekst:\n{text}\n\nZwróć JSON: {{\"core_concepts\":[]}}")
])



def extract_all_concepts(state: GraphState):
    text = state["text"]

    ner_chain = ner_prompt | llm_json | JsonOutputParser(pydantic_object=EntitiesOutput)
    events_chain = events_prompt | llm_json | JsonOutputParser(pydantic_object=EventsOutput)
    concepts_chain = concepts_prompt | llm_json | JsonOutputParser(pydantic_object=CoreConceptsOutput)

    try:
        ner_res = ner_chain.invoke({"text": text}).dict()
    except:
        ner_res = {"persons": [], "organizations": [], "locations": []}

    try:
        events_res = events_chain.invoke({"text": text}).dict()
    except:
        events_res = {"events": [], "time_expressions": []}

    try:
        concepts_res = concepts_chain.invoke({"text": text}).core_concepts
    except:
        concepts_res = []

    return {
        "entities": ner_res,
        "events": events_res,
        "core_concepts": concepts_res
    }



qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Wygeneruj pytania i odpowiedzi dotyczące relacji między pojęciami. "
     "Każde pytanie musi dotyczyć co najmniej dwóch pojęć."),
    ("user",
    """
Tekst:
{text}

Pojęcia:
{concepts}

Wygeneruj JSON:
{{
  "entries": [
    {{
      "concept": "<pojęcie>",
      "qa": [
        {{"q": "...", "a": "..."}}
      ]
    }}
  ]
}}""")
])

def generate_qa_node(state: GraphState):
    text = state["text"]

    all_concepts = (
        state["entities"]["persons"]
        + state["entities"]["organizations"]
        + state["entities"]["locations"]
        + state["events"]["events"]
        + state["core_concepts"]
    )

    chain = qa_prompt | llm_json | JsonOutputParser(pydantic_object=QAGenerationOutput)

    try:
        output = chain.invoke({
            "text": text,
            "concepts": json.dumps(all_concepts, ensure_ascii=False)
        }).dict()
    except:
        output = {"entries": []}

    return {"qa_pairs": output["entries"]}


triples_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Buduj relacje tylko między pojęciami z listy. "
     "Używaj prostych predykatów. Nie dodawaj wiedzy spoza tekstu."),
    ("user",
    """
Tekst:
{text}

Pojęcia:
{concepts}

Pytania i odpowiedzi:
{qa}

Zwróć JSON:
{{"triples":[{{"subject":"","predicate":"","object":""}}]}}
    """)
])

def extract_triples_node(state: GraphState):
    text = state["text"]

    all_concepts = (
        state["entities"]["persons"]
        + state["entities"]["organizations"]
        + state["entities"]["locations"]
        + state["events"]["events"]
        + state["core_concepts"]
    )

    chain = triples_prompt | llm_json | JsonOutputParser(pydantic_object=TriplesOutput)

    try:
        output = chain.invoke({
            "text": text,
            "concepts": json.dumps(all_concepts, ensure_ascii=False),
            "qa": json.dumps(state["qa_pairs"], ensure_ascii=False)
        }).dict()
    except:
        output = {"triples": []}

    validated = [
        t for t in output["triples"]
        if t["subject"] in all_concepts and t["object"] in all_concepts
    ]

    return {"triples": validated}


def refine_triples_node(state: GraphState):
    concepts = (
        state["entities"]["persons"]
        + state["entities"]["organizations"]
        + state["entities"]["locations"]
        + state["events"]["events"]
        + state["core_concepts"]
    )

    present = set()
    for t in state["triples"]:
        present.add(t["subject"])
        present.add(t["object"])

    missing = [c for c in concepts if c not in present]

    if not missing:
        return {}

    refinement_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Uzupełnij brakujące relacje między podanymi pojęciami."),
        ("user",
        """
Brakujące pojęcia:
{missing}

Wszystkie pojęcia:
{concepts}

Zwróć JSON:
{{"triples":[{{"subject":"","predicate":"","object":""}}]}}
        """)
    ])

    chain = refinement_prompt | llm_json | JsonOutputParser(pydantic_object=TriplesOutput)

    try:
        output = chain.invoke({
            "missing": json.dumps(missing, ensure_ascii=False),
            "concepts": json.dumps(concepts, ensure_ascii=False)
        }).dict()
    except:
        output = {"triples": []}

    return {"triples": state["triples"] + output["triples"]}


workflow = StateGraph(GraphState)

workflow.add_node("extract_concepts", extract_all_concepts)
workflow.add_node("generate_qa", generate_qa_node)
workflow.add_node("extract_triples", extract_triples_node)
workflow.add_node("refine_triples", refine_triples_node)

workflow.set_entry_point("extract_concepts")
workflow.add_edge("extract_concepts", "generate_qa")
workflow.add_edge("generate_qa", "extract_triples")
workflow.add_edge("extract_triples", "refine_triples")
workflow.add_edge("refine_triples", END)

app = workflow.compile()


if __name__ == "__main__":
    from pprint import pprint

    documents = [
        """Donald Trump nie ogranicza się do przebudowy Białego Domu. Zmienia też to, jak Amerykanie patrzą na świat. Wenezuela liczy się dla niego bardziej niż Ukraina, bo to z Ameryki Łacińskiej pochodzą największe kłopoty dzisiejszych Stanów: nielegalna imigracja i przemyt narkotyków. Ważny jest również Pacyfik i rywalizacja z Chinami o hegemonię w tym regionie. Trump uważa, że USA traciły na porządku, który powstał po zakończeniu zimnej wojny. To uproszczenie, bo jako główny gwarant tego porządku Amerykanie sami wiele na nim zyskali. Z czasem jednak jego wielkim beneficjentem stali się Chińczycy. Wciąż trwa spór o to, czy przyjęcie ChRL do WTO było dobrą decyzją. Jednocześnie Ameryka wiele zyskiwała, między innymi dzięki temu, że dolar pozostawał walutą rezerwową świata.""",
        """Donald Trump od początku traktował konflikt w Ukrainie jako nieswoją wojnę. Nie miał cierpliwości ani zrozumienia dla ukraińskich racji. Błędnie zakładał też, że zdoła oderwać Rosję od sojuszu z Chinami. Tymczasem oba kraje mocno zacieśniły współpracę gospodarczą, a Chiny dostarczają dziś Rosjanom maszyny i technologie, na które monopol miał kiedyś Zachód. Moskwa i Pekin pamiętają czasy, gdy były podzielone. Putin i Xi są zgodni, że rozpad Związku Radzieckiego nie był koniecznością, tylko skutkiem błędów Kremla. Rosję i Chiny dzieli przepaść kulturalna, bo Rosjanie nie chcą być Chińczykami, ale to akurat ułatwia relacje. Współpraca z Europą grozi w oczach Putina zarażeniem rosyjskiego społeczeństwa „demokratycznym wirusem”, a w kontaktach z Chinami takiego ryzyka nie ma. Rosja może jednak wpaść w nieodwracalną zależność od Chin. Decydując się na inwazję na Ukrainę, Putin chciał przede wszystkim zasilić ludność Rosji dziesiątkami milionów Słowian, co było dla niego ważniejsze od zdobyczy terytorialnych. W tle są obawy przed chińską dominacją."""
    ]

    full_text = "\n".join(documents)

    state = {
        "text": full_text,
        "entities": {"persons": [], "organizations": [], "locations": []},
        "events": {"events": [], "time_expressions": []},
        "core_concepts": [],
        "qa_pairs": [],
        "triples": []
    }

    result = app.invoke(state)

    pprint(result)
