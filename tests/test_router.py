import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from router import classify_intent


def test_classifies_list_query():
    result = classify_intent("Zeige mir alle aktuellen Ausschreibungen zur Wahlunterlagen-Erstellung.")
    assert result == "LIST"


def test_classifies_specific_question_with_city():
    result = classify_intent("Was sind die Anforderungen für Köln?")
    assert result == "SPECIFIC"


def test_specific_question_not_misclassified_as_list():
    """This is the exact bug that broke the app earlier: the word
    'wahlunterlagen' being in the list-trigger keywords meant almost
    every specific question got wrongly routed to LIST. This test
    locks in that a specific, city-scoped question about the same
    topic word is correctly classified as SPECIFIC, not LIST."""
    result = classify_intent("Bis wann müssen die Wahlunterlagen zugestellt werden?")
    assert result == "SPECIFIC"


def test_classifies_comparison_query():
    result = classify_intent("Vergleiche alle Ausschreibungen miteinander.")
    assert result == "COMPARE"


def test_classifies_summary_query():
    result = classify_intent("Fasse die Ausschreibung für Köln zusammen.")
    assert result == "SUMMARY"