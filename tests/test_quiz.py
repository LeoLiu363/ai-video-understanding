"""测验判分与公开卷剥离。"""
from __future__ import annotations

from vedioai.quiz import grade_quiz, public_quiz


def test_quiz_grade_and_public_strip_answers():
    payload = {
        "quiz_id": "abcd",
        "video_id": "v1",
        "title": "demo",
        "count": 2,
        "questions": [
            {
                "id": "q1",
                "stem": "1+1?",
                "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "answer": "B",
                "explanation": "算术",
                "start_ms": 1000,
            },
            {
                "id": "q2",
                "stem": "颜色?",
                "options": {"A": "红", "B": "蓝", "C": "绿", "D": "黄"},
                "answer": "A",
                "explanation": "课里说红",
                "start_ms": 0,
            },
        ],
    }
    pub = public_quiz(payload)
    assert "answer" not in pub["questions"][0]
    assert "explanation" not in pub["questions"][0]
    assert pub["questions"][0]["options"]["B"] == "2"

    result = grade_quiz(payload, {"q1": "B", "q2": "C"})
    assert result["correct"] == 1
    assert result["total"] == 2
    assert result["details"][0]["correct"] is True
    assert result["details"][1]["correct"] is False
    assert result["details"][1]["answer"] == "A"
    assert "算术" in result["details"][0]["explanation"]
