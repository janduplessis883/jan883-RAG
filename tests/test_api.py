from fastapi.testclient import TestClient

from api import create_app


class FakeDatabase:
    def get_stats(self):
        return {"source_count": 2, "chunk_count": 4, "embedding_count": 4}


class FakeOllama:
    def list_models(self):
        return ["answer-model"]


class FakeChat:
    config = {"ollama": {"default_answer_model": "answer-model"}}
    ollama = FakeOllama()

    def generate_related_questions(self, question):
        return [f"Related to {question}"]

    def retrieve_sources(self, **kwargs):
        return [{"source_id": 1, "title": "Notes", "text": "A useful fact."}]

    def answer(self, **kwargs):
        return {"answer": "A grounded answer [S1]."}


class FakeRetrieval:
    def search(self, **kwargs):
        return [{"source_id": 1, "title": "Notes", "text": "A useful fact."}]


def client():
    return TestClient(create_app((None, FakeDatabase(), FakeRetrieval(), FakeChat())))


def test_health():
    response = client().get("/health")

    assert response.status_code == 200
    assert response.json()["omlx"]["available"] is True
    assert response.json()["database"]["source_count"] == 2


def test_search():
    response = client().post("/search", json={"query": "useful fact", "limit": 5})

    assert response.status_code == 200
    assert response.json()["results"][0]["title"] == "Notes"


def test_chat():
    response = client().post("/chat", json={"question": "What is useful?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "A grounded answer [S1]."
    assert response.json()["related_questions"] == ["Related to What is useful?"]
