"""Integration test: human feedback pipeline via FastAPI."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestFeedbackPipeline:
    def test_submit_and_read_feedback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Submit feedback via API and verify it's stored."""
        # Redirect feedback storage to tmp dir
        feedback_file = tmp_path / "human_labels.jsonl"
        monkeypatch.setattr("evaluate.FEEDBACK_FILE", feedback_file)
        monkeypatch.setattr("evaluate.FEEDBACK_DIR", tmp_path)

        from evaluate import app

        client = TestClient(app)

        # Submit feedback
        response = client.post(
            "/feedback/evaluate",
            json={"episode_id": 1, "score": 4, "evaluator_id": "tester"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["total_labels"] >= 1

        # Check stats
        response = client.get("/feedback/stats")
        assert response.status_code == 200
        stats = response.json()
        assert stats["total_labels"] >= 1
        assert stats["average_score"] > 0

    def test_invalid_score_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scores outside 1-5 should be rejected."""
        monkeypatch.setattr("evaluate.FEEDBACK_FILE", tmp_path / "labels.jsonl")
        monkeypatch.setattr("evaluate.FEEDBACK_DIR", tmp_path)

        from evaluate import app

        client = TestClient(app)

        response = client.post(
            "/feedback/evaluate",
            json={"episode_id": 1, "score": 10, "evaluator_id": "tester"},
        )
        assert response.status_code == 422  # Validation error
