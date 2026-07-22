import pytest

from app.database import session as session_module


class FakeSession:
    def __init__(self):
        self.rollback_calls = 0
        self.close_calls = 0

    def rollback(self):
        self.rollback_calls += 1

    def close(self):
        self.close_calls += 1


def test_get_db_closes_session_after_success(monkeypatch):
    db = FakeSession()
    monkeypatch.setattr(session_module, "SessionLocal", lambda: db)

    dependency = session_module.get_db()
    assert next(dependency) is db
    dependency.close()

    assert db.rollback_calls == 0
    assert db.close_calls == 1


def test_get_db_rolls_back_and_closes_session_after_error(monkeypatch):
    db = FakeSession()
    monkeypatch.setattr(session_module, "SessionLocal", lambda: db)

    dependency = session_module.get_db()
    assert next(dependency) is db

    with pytest.raises(RuntimeError):
        dependency.throw(RuntimeError("request failed"))

    assert db.rollback_calls == 1
    assert db.close_calls == 1
