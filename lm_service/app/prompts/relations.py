from typing import Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
from app.models.graph_dto import UnstructuredRelation


examples = [
    {
        "text": "Szef resortu minister spraw zagranicznych Chin Wang Yi wzywa obie walczące strony konfliktu, by wykorzystały każdą okazję do jak najszybszego rozpoczęcia rozmów pokojowych – przekazało ministerstwo w oświadczeniu po rozmowie.",
        "head": "Wang Yi",
        "head_type": "Osoba",
        "relation": "WZYWA_DO",
        "tail": "Rozmowy pokojowe",
        "tail_type": "Wydarzenie",
    },
    {
        "text": "Szef resortu minister spraw zagranicznych Chin Wang Yi wzywa obie walczące strony konfliktu, by wykorzystały każdą okazję do jak najszybszego rozpoczęcia rozmów pokojowych – przekazało ministerstwo w oświadczeniu po rozmowie.",
        "head": "Wang Yi",
        "head_type": "Osoba",
        "relation": "PEŁNI_FUNKCJĘ_W",
        "tail": "Ministerstwo Spraw Zagranicznych Chin",
        "tail_type": "Instytucja",
    },
    {
        "text": "Rozmowy są zawsze lepsze niż walka – powiedział Wang, cytowany przez Al Jazeerę, dodając, że wszystkie palące kwestie należy rozwiązywać poprzez dialog i negocjacje, a nie przy użyciu siły.",
        "head": "Wang Yi",
        "head_type": "Osoba",
        "relation": "CYTOWANY_PRZEZ",
        "tail": "Al Jazeera",
        "tail_type": "Medium",
    },
    {
        "text": "Rozmowy są zawsze lepsze niż walka – powiedział Wang, cytowany przez Al Jazeerę, dodając, że wszystkie palące kwestie należy rozwiązywać poprzez dialog i negocjacje, a nie przy użyciu siły.",
        "head": "Wang Yi",
        "head_type": "Osoba",
        "relation": "POPIERA",
        "tail": "Dialog i negocjacje",
        "tail_type": "Koncepcja",
    },
    {
        "text": "Adam Nowakowski, główny analityk w banku Pekao.\n"
                "Instytucja w najnowszym biuletynie podkreśla, że nakłady na odnawialne źródła energii "
                "w 2024 r. gwałtownie wzrosły - w ujęciu rocznym o 25,4 proc. - co wynika z realizacji "
                "nowych projektów farm wiatrowych na Bałtyku.",
        "head": "Nakłady na odnawialne źródła energii",
        "head_type": "Zjawisko",
        "relation": "WZROSŁY_O",
        "tail": "25,4%",
        "tail_type": "Wskaźnik",
    },
    {
        "text": "Adam Nowakowski, główny analityk w banku Pekao.\n"
                "Instytucja w najnowszym biuletynie podkreśla, że nakłady na odnawialne źródła energii "
                "w 2024 r. gwałtownie wzrosły - w ujęciu rocznym o 25,4 proc. - co wynika z realizacji "
                "nowych projektów farm wiatrowych na Bałtyku.",
        "head": "25,4%",
        "head_type": "Wskaźnik",
        "relation": "DOTYCZY",
        "tail": "2024",
        "tail_type": "Rok",
    },
    {
        "text": "Adam Nowakowski, główny analityk w banku Pekao.\n"
                "Instytucja w najnowszym biuletynie podkreśla, że nakłady na odnawialne źródła energii "
                "w 2024 r. gwałtownie wzrosły - w ujęciu rocznym o 25,4 proc. - co wynika z realizacji "
                "nowych projektów farm wiatrowych na Bałtyku.",
        "head": "Adam Nowakowski",
        "head_type": "Osoba",
        "relation": "PEŁNI_FUNKCJĘ_W",
        "tail": "Bank Pekao",
        "tail_type": "Firma",
    },
    {
        "text": "Adam Nowakowski, główny analityk w banku Pekao.\n"
                "Instytucja w najnowszym biuletynie podkreśla, że nakłady na odnawialne źródła energii "
                "w 2024 r. gwałtownie wzrosły - w ujęciu rocznym o 25,4 proc. - co wynika z realizacji "
                "nowych projektów farm wiatrowych na Bałtyku.",
        "head": "Adam Nowakowski",
        "head_type": "Osoba",
        "relation": "WYPOWIEDZIAŁ_SIĘ_O",
        "tail": "Nakłady na odnawialne źródła energii",
        "tail_type": "Zjawisko",
    },
]

SYSTEM_MSG = """You are a top-tier algorithm designed for extracting information in structured formats to build a dense knowledge graph rather than a minimal set of relations. Your task is to identify relations from a given text requested with the user prompt.

        You must generate the output in a JSON format containing a list with JSON objects. Each object should have the keys: "head", "head_type", "relation", "tail", and "tail_type".
        The "head" should preferably use one of the provided entities, but may include additional entities if needed.
        The "head_type" key must contain the type of the head entity.
        The "relation" key must contain the type of relationship between the "head" and the "tail.
        The "tail" key must represent the text of an entity which is the tail of the relation and the "tail_type" key must contain the type of the tail entity. The "tail" should preferably use one of the provided entities, but may include additional entities if needed.

        Maintain entity consistency:
        When using entities, it's vital to ensure consistency. If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), always use the identifier from provided list of entities for that entity.
        The knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
        
        Maximize coverage: 
        Connect as many entities as possible with valid relations. It is better to include more valid relations than to miss potential ones. Ensure that most entities are used in at least one relation. Avoid leaving entities disconnected.
        
        Decompress relations:
        Never encode multiple facts in a single relation. Split complex relations into multiple simpler ones. Prefer multiple clear relations over one overloaded relation.
        All specific details (numbers, locations, time, context) must be represented as separate nodes and relations, not embedded in the relation name.

        IMPORTANT NOTES:
        - ALL values of fields: "head_type", "relation" and "tail_type" MUST be in POLISH.
        - Do not add any explanations or additional text.
        - Return ONLY a list of JSON objects."""


def build_prompt(nodes: str, rels: Optional[str] = None) -> ChatPromptTemplate:
    parser = JsonOutputParser(pydantic_object=UnstructuredRelation)
    user_string_parts = [
        f"""Based on the following examples, extract as many relations as possible from the provided text (high recall). Generate multiple relations per entity whenever possible. It is better to include more valid relations than to miss potential ones.
        When a statement in the text expresses a complex or multi-part fact, decompose it into multiple simple relations. Each relation must be atomic, but multiple relations should be created to fully represent all information in the text.""",
        f"""Use the following entities. Try to connect as many entities as possible. Prefer using the provided entities, but you may introduce additional entities if necessary to form valid relations:
        # ENTITIES:\n{nodes}""" if nodes else "",
        f"""Use the following predefined relations, avoid using other relation that is not listed below:\n# RELATIONS:\n{rels}""" if rels else "",
        """Below are a number of examples of texts and their extracted entities and relationships.
    
        Examples:
        {examples}
    
        For the following text, extract entities and relations as in the provided example.
        {format_instructions}
        
        Before finishing, verify:
        - most entities are connected,
        - relations are not overly compressed,
        - the graph captures the majority of facts from the text,
        - there are no compressed relations (each relation is properly decompressed into multiple simple relations).
    
        Text: {input}"""
    ]

    USER_MSG = HumanMessagePromptTemplate(prompt=PromptTemplate(template="\n".join(filter(None, user_string_parts)),
                                                                input_variables=["input"], partial_variables={"format_instructions": parser.get_format_instructions(),
                                                                                                              "nodes": nodes, "rels": rels,
                                                                                                              "examples": examples}))
    return ChatPromptTemplate.from_messages([SYSTEM_MSG, USER_MSG])
