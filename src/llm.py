import os
import json
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from flashrank import Ranker
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
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

def detect_md_file_by_city_or_title(query: str) -> str | None:
    try:
        with open("tenders_index.json", "r", encoding="utf-8") as f:
            tenders = json.load(f)
        query = query.lower()
        for tender in tenders:
            if "city" in tender and tender["city"] and tender["city"].lower() in query:
                return tender.get("md_file")
        for tender in tenders:
            if "title" in tender and tender["title"].lower() in query:
                return tender.get("md_file")
    except Exception as e:
        print("Error detecting file:", e)
    return None

def answer_query(question: str):
    matched_file = detect_md_file_by_city_or_title(question)
    db = Chroma(persist_directory=DB_DIR, embedding_function=EMBEDDINGS)

    if matched_file:
        base_retriever = db.as_retriever(search_kwargs={"k": 20, "filter": {"source": matched_file}})
    else:
        base_retriever = db.as_retriever(search_kwargs={"k": 20})

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
        return_source_documents=False,
    )

    return qa.run(question)

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
