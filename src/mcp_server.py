import sys
import os
import json
sys.path.append(os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP
from llm import answer_query
from fetch_tenders import fetch_and_process, build_vector_store
from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP("TenderBot")

INDEX_FILE = os.path.join(os.path.dirname(__file__), "..", "tenders_index.json")

def _format_tender_list(limit: int = 10) -> str:
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            tenders = json.load(f)
        tenders = sorted(tenders, key=lambda t: t.get("deadline") or "", reverse=False)
        result = ""
        for t in tenders[:limit]:
            result += (
                f"### {t.get('title')}\n"
                f"- **Stadt:** {t.get('city', '—')}\n"
                f"- **Frist:** {t.get('deadline', '—')}\n"
                f"- **Link:** {t.get('link')}\n\n"
            )
        return result or "Keine aktuellen Ausschreibungen gefunden."
    except Exception as e:
        return f"Fehler: {e}"

def _detect_matching_files(query: str) -> list:
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            tenders = json.load(f)
        query = query.lower()
        matched = []
        for tender in tenders:
            if (tender.get("city") and tender["city"].lower() in query) or \
               (tender.get("title") and tender["title"].lower() in query):
                if "md_file" in tender:
                    matched.append(tender["md_file"])
        return matched
    except Exception:
        return []

@mcp.tool()
def search_tenders(question: str) -> str:
    """Answer any question about public tenders using RAG."""
    return answer_query(question)

@mcp.tool()
def list_all_tenders(limit: int = 10) -> str:
    """List all available tenders sorted by deadline."""
    return _format_tender_list(limit=limit)

@mcp.tool()
def get_tender_by_city(city: str) -> str:
    """Get tender details for a specific city."""
    question = f"Welche Ausschreibungen gibt es für {city}?"
    return answer_query(question)

@mcp.tool()
def check_deadlines() -> str:
    """Get all upcoming tender deadlines."""
    return _format_tender_list(limit=20)

@mcp.tool()
def fetch_latest_tenders() -> str:
    """Fetch latest tenders from service.bund.de and update vector database."""
    try:
        fetch_and_process()
        build_vector_store()
        return "Successfully fetched and indexed latest tenders."
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
def find_matching_tenders(query: str) -> str:
    """Find which tender documents match a query."""
    matched = _detect_matching_files(query)
    if matched:
        return f"Found {len(matched)} matching tender(s): {', '.join(matched)}"
    return "No specific tenders matched."

@mcp.tool()
def summarize_tender(city: str) -> str:
    """Generate a summary of key requirements 
    for a specific tender"""

@mcp.tool()  
def compare_tenders() -> str:
    """Compare all available tenders side by side
    highlighting differences in deadlines and requirements"""

if __name__ == "__main__":
    mcp.run(transport="stdio")