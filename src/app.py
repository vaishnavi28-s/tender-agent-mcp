import os
import json
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from flashrank import Ranker
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
DB_DIR = "tender_vector_db"
EMBEDDINGS = OpenAIEmbeddings()
LLM = ChatOpenAI(model="gpt-4", temperature=0)

prompt = PromptTemplate.from_template("""
Du bist ein hilfreicher Assistent für öffentliche Ausschreibungen in Deutschland.
Nutze ausschließlich den folgenden Kontext, um präzise Antworten zu geben. Antworte nur auf Deutsch.
Wenn keine Informationen im Kontext gefunden wurden, gib klar an, dass keine Daten verfügbar sind.

<context>
{context}
</context>

Frage: {question}
Antwort:
""")

def detect_matching_files(query: str) -> list[str]:
    try:
        with open("tenders_index.json", "r", encoding="utf-8") as f:
            tenders = json.load(f)
        query = query.lower()
        matched = []
        for tender in tenders:
            if ("city" in tender and tender["city"] and tender["city"].lower() in query) or \
               ("title" in tender and tender["title"].lower() in query):
                if "md_file" in tender:
                    matched.append(tender["md_file"])
        return matched
    except Exception:
        return []

def format_tender_list(limit=10):
    try:
        with open("tenders_index.json", "r", encoding="utf-8") as f:
            tenders = json.load(f)

        tenders = sorted(tenders, key=lambda t: t.get("deadline") or "", reverse=False)
        result = ""
        for t in tenders[:limit]:
            result += (
                f"### {t.get('title')}\n"
                f"- **Stadt:** {t.get('city', '—')}\n"
                f"- **Frist:** {t.get('deadline', '—')}\n"
                f"- **Link:** [{t.get('link')}]({t.get('link')})\n"
                f"- **Bekanntmachung:** "
                f"{('[' + t.get('announcement_url') + '](' + t.get('announcement_url') + ')') if t.get('announcement_url') else '—'}\n\n"
            )
        return result or "Keine aktuellen Ausschreibungen gefunden."
    except Exception as e:
        return f"Fehler beim Laden der Ausschreibungen: {e}"

def answer_query(question: str) -> str:
    list_keywords = ["zeige mir alle", "alle wahl ausschreibungen", "list wahl tenders", "liste aller", "zeige liste", "alle tenders", "list all tenders", "liste tenders", "list", "liste", "Zeige", "list wahlunterlagen", "list wahlunterlagen", "wahlunterlagen", "wahlunterlagen tenders", "wahlunterlagen tenders liste"]

    if any(word in question.lower() for word in list_keywords):
        return format_tender_list()

    matched_files = detect_matching_files(question)
    db = Chroma(persist_directory=DB_DIR, embedding_function=EMBEDDINGS)

    if matched_files:
        base_retriever = db.as_retriever(search_kwargs={"k": 20, "filter": {"source": {"$in": matched_files}}})
    else:
        base_retriever = db.as_retriever(search_kwargs={"k": 10})

  
    flashrank_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    reranker = FlashrankRerank(client=flashrank_client, top_n=5)

    retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=reranker
    )

    qa = RetrievalQA.from_chain_type(
        llm=LLM,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    result = qa.invoke({"query": question})

    def filter_chunks_with_keywords(docs, query):
        keywords = [kw.strip() for kw in query.lower().split()]
        return [doc for doc in docs if any(kw in doc.page_content.lower() for kw in keywords)]

    filtered_docs = filter_chunks_with_keywords(result["source_documents"], question)

    if not filtered_docs:
        return "Keine spezifischen Informationen zu deiner Frage gefunden."

    return result["result"]

st.set_page_config(page_title="TenderBot", layout="wide")
st.title("TenderBot - Ausschreibungsfragen stellen")

if "chat" not in st.session_state:
    st.session_state.chat = []

st.markdown("#### Beispiel-Fragen:")
st.markdown("- Zeige mir alle aktuellen Ausschreibungen zur Wahlunterlagen-Erstellung.")
st.markdown("- Welche Leistungen umfasst der Auftrag der Stadt Köln zur Scandienstleistung für die Kommunalwahl 2025?")

user_input = st.chat_input("Frage eingeben (z. B. Welche Unterlagen muss ich in Münster einreichen?)")
if user_input:
    with st.spinner("Antwort wird generiert..."):
        try:
            response = answer_query(user_input)
            st.session_state.chat.insert(0, ("Bot", response))
            st.session_state.chat.insert(0, ("You", user_input))
        except Exception as e:
            st.session_state.chat.insert(0, ("Bot", f"Fehler: {e}"))

if st.session_state.chat:
    st.divider()
    for speaker, message in st.session_state.chat:  # Now newest at top
        if speaker.startswith("You"):
            st.markdown(f"**{speaker}:** {message}")
        else:
            st.markdown(f"**{speaker}:**")
            st.markdown(message)
