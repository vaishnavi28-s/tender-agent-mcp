from dotenv import load_dotenv
load_dotenv()
import os
from langchain_groq import ChatGroq



VALID_CATEGORIES = {"LIST", "SUMMARY", "COMPARE", "SPECIFIC"}

_classifier_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

CLASSIFY_PROMPT = """Classify the following question about German public procurement tenders into exactly one category.

Categories:
- LIST: asks to see/browse all tenders, no specific detail needed. Examples: "zeige mir alle", "list all tenders"
- SUMMARY: asks for a summary/overview of one tender. Examples: "Zusammenfassung für Magdeburg", "summarize the Cologne tender"
- COMPARE: asks to compare multiple tenders. Examples: "vergleiche alle Ausschreibungen", "compare the tenders"
- SPECIFIC: asks a specific factual question. Examples: "welche Anforderungen gibt es für Halle", "what is the deadline for Cologne"

Respond with exactly one word: LIST, SUMMARY, COMPARE, or SPECIFIC. No punctuation, no explanation.

Question: {question}
"""

def classify_intent(question: str) -> str:
    try:
        response = _classifier_llm.invoke(CLASSIFY_PROMPT.format(question=question))
        raw = response.content.strip().upper()

        for category in VALID_CATEGORIES:
            if category in raw:
                return category

        print(f"CLASSIFIER WARNING: unrecognized response '{raw}', defaulting to SPECIFIC")
        return "SPECIFIC"

    except Exception as e:
        print(f"CLASSIFIER ERROR: {e}")
        return "SPECIFIC"