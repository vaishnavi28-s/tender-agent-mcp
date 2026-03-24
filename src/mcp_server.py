"""
TenderBot MCP Server
Features:
- Smart tool routing (avoids unnecessary LLM calls)
- Structured audit logging with latency tracking
- Input validation with Pydantic
- Per-tool rate limiting
- Error handling per exception type
- Health check with server diagnostics
- Async tools for non-blocking execution
- Token-efficient query classification
"""

import sys
import os
import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

sys.path.append(os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

load_dotenv()


# Logging setup 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),  # MCP requires stderr not stdout
        logging.FileHandler("tenderbot_audit.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("tenderbot")


def _audit_log(tool: str, input_data: dict, result: str, latency_ms: float, status: str = "success"):
    """Structured audit log entry for every tool call."""
    logger.info(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "tool": tool,
        "input": input_data,
        "status": status,
        "latency_ms": round(latency_ms, 2),
        "output_length": len(result),
    }))



# Rate limiter

class RateLimiter:
    """Simple in-memory rate limiter per tool."""

    def __init__(self):
        self._calls: dict = defaultdict(list)
        self._limits = {
            "search_tenders": (10, 60),       # 10 calls per 60 seconds
            "list_all_tenders": (20, 60),      # 20 calls per 60 seconds
            "get_tender_by_city": (10, 60),
            "check_deadlines": (20, 60),
            "fetch_latest_tenders": (3, 3600), # 3 calls per hour (expensive)
            "find_matching_tenders": (15, 60),
            "summarize_tender": (5, 60),       # LLM heavy
            "compare_tenders": (3, 60),        # LLM heavy
            "health_check": (60, 60),
        }

    def is_allowed(self, tool: str) -> tuple[bool, str]:
        limit, window = self._limits.get(tool, (20, 60))
        now = time.time()
        calls = self._calls[tool]

        # Remove expired calls
        self._calls[tool] = [t for t in calls if now - t < window]

        if len(self._calls[tool]) >= limit:
            reset_in = int(window - (now - self._calls[tool][0]))
            return False, f"Rate limit exceeded for {tool}. Try again in {reset_in}s."

        self._calls[tool].append(now)
        return True, ""


rate_limiter = RateLimiter()
SERVER_START_TIME = datetime.utcnow()


# Input validation models

class QuestionInput(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)

    @validator("question")
    def sanitize_question(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        return v


class CityInput(BaseModel):
    city: str = Field(..., min_length=2, max_length=100)

    @validator("city")
    def sanitize_city(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("City cannot be empty")
        return v


class LimitInput(BaseModel):
    limit: int = Field(default=10, ge=1, le=50)



# Saves tokens

def _is_list_query(text: str) -> bool:
    """Returns True if query just needs a list — no LLM needed."""
    keywords = [
        "list", "show all", "show me all", "alle ausschreibungen",
        "zeige alle", "zeige mir alle", "all tenders", "what tenders",
        "welche ausschreibungen", "list all", "liste alle"
    ]
    t = text.lower().strip()
    return any(kw in t for kw in keywords)


def _needs_summary(text: str) -> bool:
    keywords = ["summarize", "summary", "zusammenfassung", "überblick", "zusammenfassen"]
    t = text.lower()
    return any(kw in t for kw in keywords)


def _needs_comparison(text: str) -> bool:
    keywords = ["compare", "comparison", "vergleich", "vergleiche", "unterschied", "difference", " vs "]
    t = text.lower()
    return any(kw in t for kw in keywords)


def _extract_city(text: str) -> Optional[str]:
    """Extract city name from query text."""
    cities = ["köln", "koeln", "cologne", "münchen", "munich", "berlin",
              "hamburg", "magdeburg", "halle", "mannheim", "frankfurt",
              "stuttgart", "düsseldorf", "dortmund", "essen", "münster"]
    t = text.lower()
    for city in cities:
        if city in t:
            return city.capitalize()
    return None


# Helper functions
INDEX_FILE = os.path.join(os.path.dirname(__file__), "..", "tenders_index.json")


def _load_index() -> list:
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_tender_list(limit: int = 10) -> str:
    try:
        tenders = _load_index()
        tenders = sorted(tenders, key=lambda t: t.get("deadline") or "", reverse=False)
        result = ""
        for t in tenders[:limit]:
            result += (
                f"### {t.get('title', 'Unknown')}\n"
                f"- **Stadt:** {t.get('city', '—')}\n"
                f"- **Frist:** {t.get('deadline', '—')}\n"
                f"- **Link:** {t.get('link', '—')}\n\n"
            )
        return result or "Keine aktuellen Ausschreibungen gefunden."
    except FileNotFoundError:
        return "Tender index not found. Run fetch_latest_tenders first."
    except Exception as e:
        logger.error(f"_format_tender_list error: {e}")
        return f"Error loading tender list: {e}"


def _detect_matching_files(query: str) -> list:
    try:
        tenders = _load_index()
        q = query.lower()
        return [
            t["md_file"]
            for t in tenders
            if "md_file" in t and (
                (t.get("city") and t["city"].lower() in q) or
                (t.get("title") and t["title"].lower() in q)
            )
        ]
    except Exception:
        return []


def _check_rate_limit(tool: str) -> Optional[str]:
    allowed, msg = rate_limiter.is_allowed(tool)
    if not allowed:
        logger.warning(f"Rate limit hit: {tool}")
        return msg
    return None



# MCP Server

mcp = FastMCP(
    "TenderBot",
    instructions=(
        "TenderBot answers questions about German public procurement tenders. "
        "Use search_tenders for specific questions, list_all_tenders to browse, "
        "get_tender_by_city for location-specific queries, summarize_tender for "
        "summaries, and compare_tenders to compare multiple tenders side by side. "
        "Call fetch_latest_tenders sparingly — it triggers live web scraping."
    )
)


@mcp.tool()
async def search_tenders(question: str) -> str:
    """
    Answer any question about public tenders using RAG over indexed documents.

    Automatically routes to cheaper operations when possible:
    - List queries → returns formatted list (no LLM)
    - Summary queries → routes to summarize_tender
    - Comparison queries → routes to compare_tenders
    - Specific questions → uses RAG + LLM

    Args:
        question: Natural language query in German or English.
                  Examples: "Welche Fristen gelten für Köln?"
                            "What are the submission requirements for Magdeburg?"

    Returns:
        Grounded answer with source references, or error message if unavailable.
    """
    t0 = time.time()
    tool = "search_tenders"

    # Rate limit check
    if err := _check_rate_limit(tool):
        return err

    # Input validation
    try:
        validated = QuestionInput(question=question)
        question = validated.question
    except Exception as e:
        return f"Invalid input: {e}"

    # Smart routing to avoid unnecessary LLM calls
    if _is_list_query(question):
        result = _format_tender_list(limit=10)
        _audit_log(tool, {"question": question, "routed_to": "list"}, result, (time.time() - t0) * 1000)
        return result

    if _needs_comparison(question):
        result = await compare_tenders()
        _audit_log(tool, {"question": question, "routed_to": "compare"}, result, (time.time() - t0) * 1000)
        return result

    if _needs_summary(question):
        city = _extract_city(question)
        if city:
            result = await summarize_tender(city)
            _audit_log(tool, {"question": question, "routed_to": "summarize"}, result, (time.time() - t0) * 1000)
            return result

    # Full RAG + LLM path
    try:
        from llm import answer_query
        result = await asyncio.get_event_loop().run_in_executor(None, answer_query, question)

        if not result or result.strip() == "":
            result = "Keine relevanten Informationen gefunden. Bitte versuche eine spezifischere Frage."

        latency = (time.time() - t0) * 1000
        _audit_log(tool, {"question": question}, result, latency)
        return result

    except ConnectionError:
        return "Error: Cannot connect to OpenAI API. Check your API key and internet connection."
    except TimeoutError:
        return "Error: Request timed out. Try a shorter or simpler question."
    except Exception as e:
        logger.error(f"{tool} failed: {e}")
        _audit_log(tool, {"question": question}, str(e), (time.time() - t0) * 1000, "error")
        return f"Tool temporarily unavailable. Please try again."


@mcp.tool()
async def list_all_tenders(limit: int = 10) -> str:
    """
    List all available tenders sorted by deadline.

    Args:
        limit: Number of tenders to return (1-50, default 10).

    Returns:
        Formatted list with title, city, deadline and link for each tender.
    """
    t0 = time.time()
    tool = "list_all_tenders"

    if err := _check_rate_limit(tool):
        return err

    try:
        validated = LimitInput(limit=limit)
    except Exception as e:
        return f"Invalid input: {e}"

    result = _format_tender_list(limit=validated.limit)
    _audit_log(tool, {"limit": validated.limit}, result, (time.time() - t0) * 1000)
    return result


@mcp.tool()
async def get_tender_by_city(city: str) -> str:
    """
    Get tender details for a specific German city.

    Args:
        city: City name in German or English.
              Examples: "Köln", "München", "Magdeburg", "Berlin"

    Returns:
        Tender details including deadlines, requirements, and submission info.
    """
    t0 = time.time()
    tool = "get_tender_by_city"

    if err := _check_rate_limit(tool):
        return err

    try:
        validated = CityInput(city=city)
        city = validated.city
    except Exception as e:
        return f"Invalid input: {e}"

    try:
        from llm import answer_query
        question = f"Welche Ausschreibungen gibt es für {city}? Was sind die Fristen und Anforderungen?"
        result = await asyncio.get_event_loop().run_in_executor(None, answer_query, question)
        _audit_log(tool, {"city": city}, result, (time.time() - t0) * 1000)
        return result
    except Exception as e:
        logger.error(f"{tool} failed: {e}")
        return f"Could not retrieve tenders for {city}. Please try again."


@mcp.tool()
async def check_deadlines() -> str:
    """
    Get all upcoming tender deadlines sorted by date.

    Returns:
        All tenders sorted by deadline (earliest first) with city and link.
    """
    t0 = time.time()
    tool = "check_deadlines"

    if err := _check_rate_limit(tool):
        return err

    result = _format_tender_list(limit=20)
    _audit_log(tool, {}, result, (time.time() - t0) * 1000)
    return result


@mcp.tool()
async def fetch_latest_tenders() -> str:
    """
    Fetch the latest tenders from service.bund.de and rebuild the vector database.

    WARNING: This triggers live web scraping and OpenAI embedding calls.
    Use sparingly — rate limited to 3 calls per hour.

    Returns:
        Status message with number of tenders fetched and indexed.
    """
    t0 = time.time()
    tool = "fetch_latest_tenders"

    if err := _check_rate_limit(tool):
        return err

    try:
        from fetch_tenders import fetch_and_process, build_vector_store
        await asyncio.get_event_loop().run_in_executor(None, fetch_and_process)
        await asyncio.get_event_loop().run_in_executor(None, build_vector_store)

        # Report how many tenders were indexed
        try:
            tenders = _load_index()
            count = len(tenders)
        except Exception:
            count = "unknown"

        result = f"Successfully fetched and indexed {count} tenders from service.bund.de."
        _audit_log(tool, {}, result, (time.time() - t0) * 1000)
        return result

    except Exception as e:
        logger.error(f"{tool} failed: {e}")
        _audit_log(tool, {}, str(e), (time.time() - t0) * 1000, "error")
        return f"Fetch failed: {e}. Check internet connection and API key."


@mcp.tool()
async def find_matching_tenders(query: str) -> str:
    """
    Find which tender documents match a query by city or title keyword.

    Useful for checking what data is available before running a full search.
    Does NOT use LLM — purely keyword matching over the index.

    Args:
        query: Search term (city name or keyword from tender title).

    Returns:
        List of matching tender filenames and count.
    """
    t0 = time.time()
    tool = "find_matching_tenders"

    if err := _check_rate_limit(tool):
        return err

    try:
        validated = QuestionInput(question=query)
        query = validated.question
    except Exception as e:
        return f"Invalid input: {e}"

    matched = _detect_matching_files(query)
    if matched:
        result = f"Found {len(matched)} matching tender(s):\n" + "\n".join(f"- {m}" for m in matched)
    else:
        result = "No tenders matched. Try a different city name or keyword."

    _audit_log(tool, {"query": query}, result, (time.time() - t0) * 1000)
    return result


@mcp.tool()
async def summarize_tender(city: str) -> str:
    """
    Generate a structured summary of key requirements for a city's tender.

    Extracts: scope of work, eligibility criteria, deadlines,
    submission requirements, and contact information.

    Args:
        city: City name whose tender should be summarized.
              Examples: "Köln", "Magdeburg", "Halle"

    Returns:
        Structured summary with key tender requirements.
    """
    t0 = time.time()
    tool = "summarize_tender"

    if err := _check_rate_limit(tool):
        return err

    try:
        validated = CityInput(city=city)
        city = validated.city
    except Exception as e:
        return f"Invalid input: {e}"

    # Check if we even have data for this city before calling LLM
    matched = _detect_matching_files(city)
    if not matched:
        return f"No tender documents found for '{city}'. Available cities can be found using list_all_tenders."

    try:
        from llm import answer_query
        prompt = (
            f"Erstelle eine strukturierte Zusammenfassung der Ausschreibung für {city}. "
            f"Bitte antworte auf Deutsch und gliedere die Antwort in: "
            f"1) Leistungsumfang, 2) Anforderungen, 3) Fristen, 4) Einreichungsanforderungen."
        )
        result = await asyncio.get_event_loop().run_in_executor(None, answer_query, prompt)
        _audit_log(tool, {"city": city}, result, (time.time() - t0) * 1000)
        return result
    except Exception as e:
        logger.error(f"{tool} failed: {e}")
        return f"Could not generate summary for {city}. Please try again."


@mcp.tool()
async def compare_tenders() -> str:
    """
    Compare all available tenders side by side.

    Highlights differences in: deadlines, scope, location,
    requirements, and submission procedures.

    Returns:
        Structured comparison table of all available tenders.
    """
    t0 = time.time()
    tool = "compare_tenders"

    if err := _check_rate_limit(tool):
        return err

    try:
        tenders = _load_index()
    except FileNotFoundError:
        return "No tender data available. Run fetch_latest_tenders first."

    if len(tenders) < 2:
        return "Need at least 2 tenders to compare. Currently only 1 available."

    # Build comparison without LLM if possible (saves tokens)
    if len(tenders) <= 5:
        result = "## Tender Comparison\n\n"
        result += "| Title | City | Deadline | Link |\n"
        result += "|---|---|---|---|\n"
        for t in sorted(tenders, key=lambda x: x.get("deadline") or ""):
            result += (
                f"| {t.get('title', '—')[:50]} "
                f"| {t.get('city', '—')} "
                f"| {t.get('deadline', '—')} "
                f"| [Link]({t.get('link', '#')}) |\n"
            )
        _audit_log(tool, {"count": len(tenders)}, result, (time.time() - t0) * 1000, "table")
        return result

    # For many tenders use LLM for intelligent comparison
    try:
        from llm import answer_query
        prompt = (
            "Vergleiche alle verfügbaren Ausschreibungen und hebe die wichtigsten "
            "Unterschiede hervor: Fristen, Leistungsumfang, Anforderungen und Standorte. "
            "Antworte auf Deutsch in einer strukturierten Tabelle."
        )
        result = await asyncio.get_event_loop().run_in_executor(None, answer_query, prompt)
        _audit_log(tool, {"count": len(tenders)}, result, (time.time() - t0) * 1000)
        return result
    except Exception as e:
        logger.error(f"{tool} failed: {e}")
        return f"Comparison failed: {e}"


@mcp.tool()
async def health_check() -> str:
    """
    Check TenderBot server health and diagnostics.

    Returns server status, uptime, tool availability,
    vector database connection, and tender index stats.
    """
    uptime = datetime.utcnow() - SERVER_START_TIME
    hours, remainder = divmod(int(uptime.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)

    # Check vector DB
    try:
        from llm import EMBEDDINGS, DB_DIR
        import chromadb
        client = chromadb.PersistentClient(path=DB_DIR)
        collections = client.list_collections()
        db_status = f"connected ({len(collections)} collection(s))"
    except Exception as e:
        db_status = f"unavailable: {e}"

    # Check tender index
    try:
        tenders = _load_index()
        index_status = f"loaded ({len(tenders)} tenders)"
        next_deadline = min(
            (t.get("deadline") for t in tenders if t.get("deadline")),
            default="unknown"
        )
    except Exception:
        index_status = "not found"
        next_deadline = "unknown"

    # Check OpenAI key
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_status = "configured" if api_key.startswith("sk-") else "missing"

    result = (
        f"## TenderBot Health Check\n\n"
        f"**Status:** healthy\n"
        f"**Uptime:** {hours}h {minutes}m {seconds}s\n"
        f"**Vector DB:** {db_status}\n"
        f"**Tender Index:** {index_status}\n"
        f"**Next Deadline:** {next_deadline}\n"
        f"**OpenAI API:** {api_status}\n"
        f"**Tools Available:** 9\n"
        f"**Timestamp:** {datetime.utcnow().isoformat()}Z\n"
    )

    _audit_log("health_check", {}, result, 0)
    return result


if __name__ == "__main__":
    logger.info("TenderBot MCP Server starting...")
    mcp.run(transport="stdio")
