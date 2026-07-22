import json
from pathlib import Path


def load_questions():

    path = Path("tests/test_retriver/questions.json")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)