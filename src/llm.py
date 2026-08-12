from dotenv import load_dotenv
load_dotenv()

import os
import json
import sys
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from flashrank import Ranker, RerankRequest
from langsmith import traceable
sys.path.append(os.path.dirname(__file__))
from router import classify_intent
from formatting import format_tender_list



DB_DIR = "tender_vector_db"
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "tenders_index.json")
RELEVANCE_THRESHOLD = 0.3  # tune this empirically, see note below

EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
LLM = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=os.getenv("GROQ_API_KEY"), reasoning_format="hidden")
RERANKER = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

prompt = PromptTemplate.from_template("""
Du bist ein hilfreicher Assistent fÃ¼r Ã¶ffentliche Ausschreibungen in Deutschland.
Nutze ausschlieÃŸlich den folgenden Kontext, um prÃ¤zise Antworten zu geben. Antworte nur auf Deutsch.
Wenn keine Informationen im Kontext gefunden wurden, gib klar an, dass keine Daten verfÃ¼gbar sind.

<context>
{context}
</context>

Frage: {question}
Antwort:
""")


def detect_md_file_by_city_or_title(query: str) -> str | None:
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            tenders = json.load(f)
        query = query.lower()
        for tender in tenders:
            if tender.get("city") and tender["city"].lower() in query:
                return tender.get("md_file")
        for tender in tenders:
            if tender.get("title") and tender["title"].lower() in query:
                return tender.get("md_file")
    except Exception as e:
        print("Error detecting file:", e)
    return None


@traceable(name="answer_query")
def answer_query(question: str, history: list | None = None, return_metadata: bool = False):
    intent = classify_intent(question)

    if intent == "LIST":
        answer = format_tender_list(index_file=INDEX_PATH)
        if return_metadata:
            return {"answer": answer, "route": intent, "score": None}
        return answer

    matched_file = detect_md_file_by_city_or_title(question)

    # If nothing matched in THIS question, try carrying forward the city
    # from the most recent previous question in the conversation.
    if not matched_file and history:
        for entry in reversed(history):
            if entry.get("role") != "user":
                continue
            content = entry.get("content", [])
            if not content or not isinstance(content, list):
                continue
            prior_text = content[0].get("text", "")
            prior_match = detect_md_file_by_city_or_title(prior_text)
            if prior_match:
                matched_file = prior_match
                break

    db = Chroma(persist_directory=DB_DIR, embedding_function=EMBEDDINGS)

    if matched_file:
        base_docs = db.as_retriever(
            search_kwargs={"k": 20, "filter": {"source": matched_file}}
        ).invoke(question)
    else:
        base_docs = db.as_retriever(search_kwargs={"k": 20}).invoke(question)

    if not base_docs:
        answer = "Keine passenden Ausschreibungen gefunden."
        if return_metadata:
            return {"answer": answer, "route": intent, "score": None}
        return answer

    # Manual reranking (not via LangChain's ContextualCompressionRetriever wrapper)
    # so we get real numeric relevance scores, not just a reordered list.
    passages = [
        {"id": i, "text": doc.page_content, "meta": doc.metadata}
        for i, doc in enumerate(base_docs)
    ]
    rerank_results = RERANKER.rerank(RerankRequest(query=question, passages=passages))

    top_results = rerank_results[:5]
    top_score = top_results[0]["score"] if top_results else 0.0
    print(f"Top relevance score: {top_score}")

    # Grounding guardrail: if even the best-matching chunk is a poor match,
    # don't call the LLM at all â€” refuse honestly instead of risking a
    # confident-sounding answer built on irrelevant context.
    if top_score < RELEVANCE_THRESHOLD:
        answer = "Keine spezifischen Informationen zu deiner Frage gefunden."
        if return_metadata:
            return {"answer": answer, "route": intent, "score": round(float(top_score), 3)}
        return answer

    context = "\n\n".join(r["text"] for r in top_results)
    formatted_prompt = prompt.format(context=context, question=question)
    response = LLM.invoke(formatted_prompt)
    answer_text = response.content.strip() if response.content else "Entschuldigung, es gab ein Problem bei der Antwortgenerierung. Bitte versuche es erneut."
    if return_metadata:
        return {"answer": answer_text, "route": intent, "score": round(float(top_score), 3)}
    return answer_text


if __name__ == "__main__":
    print("Ausschreibungs-LLM gestartet. Tippe 'exit' zum Beenden.")
    while True:
        q = input("\nFrage: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        try:
            print("\nAntwort:")
            print(answer_query(q))
        except Exception as e:
            print("Fehler:", e)


