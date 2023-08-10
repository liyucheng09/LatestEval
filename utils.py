from dataclasses import dataclass

@dataclass
class Doc:
    text: str
    source: str
    entry_id: str
    
    original_passage: str = None
    passage_to_input: str = None
    
    original_sentences: list = None

    queries: list[str] = None
    answers: list[str] = None
    answers_sentences: list[(str, str)] = None
    meta_data: dict = None

    sections: list[str] = None

    def __repr__(self):
        return f"Doc({source}, {entry_id})"