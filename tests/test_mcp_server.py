"""
Tests for prod MCP server features.
Run with: pytest tests/test_mcp_server.py -v
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mcp_server import (
    _is_list_query,
    _needs_summary,
    _needs_comparison,
    _extract_city,
    _format_tender_list,
    _detect_matching_files,
    RateLimiter,
    QuestionInput,
    CityInput,
    LimitInput,
)
from pydantic import ValidationError


class TestQueryClassifier:
    """Test smart query routing — no LLM calls needed."""

    def test_list_query_english(self):
        assert _is_list_query("list all tenders") is True

    def test_list_query_german(self):
        assert _is_list_query("zeige alle ausschreibungen") is True

    def test_list_query_show_all(self):
        assert _is_list_query("show me all tenders") is True

    def test_specific_question_not_list(self):
        assert _is_list_query("What are the deadlines for Köln?") is False

    def test_summary_query_english(self):
        assert _needs_summary("summarize the Köln tender") is True

    def test_summary_query_german(self):
        assert _needs_summary("zusammenfassung der Ausschreibung") is True

    def test_comparison_query(self):
        assert _needs_comparison("compare all tenders") is True

    def test_comparison_german(self):
        assert _needs_comparison("vergleiche die Ausschreibungen") is True

    def test_regular_question_not_summary(self):
        assert _needs_summary("What is the deadline for Köln?") is False


class TestCityExtraction:
    def test_extract_koeln(self):
        assert _extract_city("Was sind die Fristen für Köln?") == "Köln"

    def test_extract_munich(self):
        assert _extract_city("tenders in Munich") == "Munich"

    def test_extract_magdeburg(self):
        assert _extract_city("Magdeburg tender deadline") == "Magdeburg"

    def test_no_city(self):
        assert _extract_city("What are the requirements?") is None


class TestInputValidation:
    def test_valid_question(self):
        q = QuestionInput(question="What are the deadlines?")
        assert q.question == "What are the deadlines?"

    def test_question_too_short(self):
        with pytest.raises(ValidationError):
            QuestionInput(question="Hi")

    def test_question_too_long(self):
        with pytest.raises(ValidationError):
            QuestionInput(question="x" * 1001)

    def test_question_stripped(self):
        q = QuestionInput(question="  valid question  ")
        assert q.question == "valid question"

    def test_valid_city(self):
        c = CityInput(city="Köln")
        assert c.city == "Köln"

    def test_city_too_short(self):
        with pytest.raises(ValidationError):
            CityInput(city="K")

    def test_valid_limit(self):
        l = LimitInput(limit=20)
        assert l.limit == 20

    def test_limit_too_high(self):
        with pytest.raises(ValidationError):
            LimitInput(limit=100)

    def test_limit_zero(self):
        with pytest.raises(ValidationError):
            LimitInput(limit=0)


class TestRateLimiter:
    def test_allows_normal_calls(self):
        limiter = RateLimiter()
        allowed, msg = limiter.is_allowed("list_all_tenders")
        assert allowed is True
        assert msg == ""

    def test_blocks_after_limit(self):
        limiter = RateLimiter()
        # fetch_latest_tenders is limited to 3 per hour
        for _ in range(3):
            limiter.is_allowed("fetch_latest_tenders")
        allowed, msg = limiter.is_allowed("fetch_latest_tenders")
        assert allowed is False
        assert "Rate limit exceeded" in msg

    def test_different_tools_independent(self):
        limiter = RateLimiter()
        # Exhaust fetch_latest_tenders
        for _ in range(3):
            limiter.is_allowed("fetch_latest_tenders")
        # list_all_tenders should still work
        allowed, _ = limiter.is_allowed("list_all_tenders")
        assert allowed is True


class TestTenderList:
    def test_format_returns_string(self):
        """_format_tender_list should always return a string."""
        result = _format_tender_list(limit=5)
        assert isinstance(result, str)

    def test_format_handles_missing_file(self, tmp_path, monkeypatch):
        """Should return error message if index not found."""
        import mcp_server
        monkeypatch.setattr(mcp_server, "INDEX_FILE", str(tmp_path / "nonexistent.json"))
        result = _format_tender_list()
        assert "not found" in result.lower() or "error" in result.lower()
