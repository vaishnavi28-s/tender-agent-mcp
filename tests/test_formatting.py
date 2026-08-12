import sys
import os
import json
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from formatting import format_tender_list


def make_temp_index(tenders):
    """Helper: writes a temporary tenders_index.json for testing, so tests
    don't depend on the real, changing data file."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(tenders, f)
    return path


def test_format_shows_all_when_under_limit():
    tenders = [
        {"title": "Tender A", "city": "Köln", "deadline": "2026-08-01", "link": "http://a", "announcement_url": None},
        {"title": "Tender B", "city": "Halle", "deadline": "2026-09-01", "link": "http://b", "announcement_url": None},
    ]
    path = make_temp_index(tenders)
    result = format_tender_list(index_file=path, limit=10)
    assert "Tender A" in result
    assert "Tender B" in result
    assert "Showing" not in result  # nothing truncated, no disclosure needed
    os.remove(path)


def test_format_shows_truncation_notice_when_over_limit():
    tenders = [
        {"title": f"Tender {i}", "city": "Köln", "deadline": "2026-08-01", "link": "http://x", "announcement_url": None}
        for i in range(15)
    ]
    path = make_temp_index(tenders)
    result = format_tender_list(index_file=path, limit=10)
    assert "Showing 10 of 15" in result
    os.remove(path)


def test_format_handles_missing_file():
    result = format_tender_list(index_file="this_file_does_not_exist.json", limit=10)
    assert "not found" in result.lower()