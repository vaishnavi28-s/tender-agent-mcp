import sys
import os
sys.path.append(os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP
from llm import answer_query
from fetch_tenders import fetch_and_process, build_vector_store
from app import format_tender_list, detect_matching_files

mcp = FastMCP("TenderBot")


@mcp.tool()
def search_tenders(question: str) -> str:
    """
    Answer any question about public tenders using RAG.
    Use this for questions like deadlines, eligibility, requirements, city-specific queries.
    """
    return answer_query(question)


@mcp.tool()
def list_all_tenders(limit: int = 10) -> str:
    """
    List all available tenders sorted by deadline.
    Returns title, city, deadline and link for each tender.
    """
    return format_tender_list(limit=limit)


@mcp.tool()
def get_tender_by_city(city: str) -> str:
    """
    Get tender details for a specific city.
    Example: get_tender_by_city("Köln") or get_tender_by_city("München")
    """
    question = f"Welche Ausschreibungen gibt es für {city}? Was sind die Fristen und Anforderungen?"
    return answer_query(question)


@mcp.tool()
def check_deadlines() -> str:
    """
    Get all upcoming tender deadlines sorted by date.
    """
    return format_tender_list(limit=20)


@mcp.tool()
def fetch_latest_tenders() -> str:
    """
    Fetch the latest tenders from service.bund.de and update the vector database.
    Use this to get fresh tender data.
    """
    try:
        fetch_and_process()
        build_vector_store()
        return "Successfully fetched and indexed latest tenders."
    except Exception as e:
        return f"Error fetching tenders: {e}"


@mcp.tool()
def find_matching_tenders(query: str) -> str:
    """
    Find which tender documents match a given query by city or title.
    Returns the list of matching tender file names.
    """
    matched = detect_matching_files(query)
    if matched:
        return f"Found {len(matched)} matching tender(s): {', '.join(matched)}"
    return "No specific tenders matched. Will search across all tenders."


if __name__ == "__main__":
    mcp.run(transport="stdio")
