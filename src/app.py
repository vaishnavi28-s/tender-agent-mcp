import sys
import os
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__)))
from llm import answer_query

st.set_page_config(page_title="TenderBot", layout="wide")
st.title("TenderBot - Ausschreibungsfragen stellen")

with st.expander("Uber dieses Projekt / About this project"):
    st.markdown("""
**Architecture:** LLM-based intent classification (LIST/SUMMARY/COMPARE/SPECIFIC) then
retrieval (ChromaDB) then reranking (FlashRank) then a grounding guardrail (calibrated
relevance threshold) then generation (Groq, openai/gpt-oss-20b).

**Grounding guardrail:** answers are refused if the reranked relevance score falls below
0.3, calibrated from real test data (a genuine no-match scored 0.049, a real match scored
0.774), not guessed.

**Also exposed via MCP:** 9 rate-limited, audit-logged tools (search, list, summarize,
compare, health check, and more). See the GitHub repo for the MCP server, Claude Desktop
integration, and full architecture writeup. This Streamlit view demos the RAG/chat layer only.

**Scope:** currently indexes "Wahlunterlagen" (election-material) tenders from
service.bund.de. Data freshness depends on the last scheduled ingestion run.
    """)

if "chat" not in st.session_state:
    st.session_state.chat = []

st.markdown("#### Beispiel-Fragen:")
st.markdown("- Zeige mir alle aktuellen Ausschreibungen zur Wahlunterlagen-Erstellung.")
st.markdown("- Welche Leistungen umfasst der Auftrag der Stadt Koln zur Scandienstleistung fur die Kommunalwahl 2025?")

user_input = st.chat_input("Frage eingeben (z. B. Welche Unterlagen muss ich in Munster einreichen?)")
if user_input:
    with st.spinner("Antwort wird generiert..."):
        try:
            result = answer_query(user_input, return_metadata=True)
            st.session_state.chat.append(("You", user_input))
            st.session_state.chat.append(("Bot", result))
        except Exception as e:
            st.session_state.chat.append(("You", user_input))
            st.session_state.chat.append(("Bot", {"answer": f"Fehler: {e}", "route": None, "score": None}))

if st.session_state.chat:
    st.divider()
    for speaker, message in st.session_state.chat:
        if speaker.startswith("You"):
            st.markdown(f"**{speaker}:** {message}")
        else:
            st.markdown(f"**{speaker}:**")
            st.markdown(message["answer"])
            caption_parts = []
            if message.get("route"):
                caption_parts.append(f"Route: `{message['route']}`")
            if message.get("score") is not None:
                caption_parts.append(f"Grounding confidence: `{message['score']}`")
            if caption_parts:
                st.caption(" - ".join(caption_parts))
