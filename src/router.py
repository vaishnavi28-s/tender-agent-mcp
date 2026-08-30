from dotenv import load_dotenv
load_dotenv()
import os
from typing import Literal
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

IntentCategory = Literal["LIST", "SUMMARY", "COMPARE", "SPECIFIC"]
_model = GroqModel(
    "openai/gpt-oss-20b",
    provider= GroqProvider(api_key=os.getenv("GROQ_API_KEY")),
)
_classifier_agent = Agent(
    _model,
    output_type=IntentCategory,   
    system_prompt="""Classify the following question about German public procurement tenders into exactly one category.

Categories:
- LIST: asks to see/browse all tenders, no specific detail needed. Examples: "zeige mir alle", "list all tenders"
- SUMMARY: asks for a summary/overview of one tender. Examples: "Zusammenfassung für Magdeburg", "summarize the Cologne tender"
- COMPARE: asks to compare multiple tenders. Examples: "vergleiche alle Ausschreibungen", "compare the tenders"
- SPECIFIC: asks a specific factual question. Examples: "welche Anforderungen gibt es für Halle", "what is the deadline for Cologne"
""",
)
def classify_intent(question: str) -> str:
    try:
        result = _classifier_agent.run_sync(question)
        return result.output
    except Exception as e:
        print(f"CLASSIFIER ERROR: {e}")
        return "SPECIFIC"
