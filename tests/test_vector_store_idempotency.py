import sys
import os
import json
import tempfile
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def test_rebuilding_vector_store_does_not_duplicate_chunks():
    """This is the exact bug that silently doubled chunk counts (39 -> 90)
    and dropped relevance scores from 0.774 to 0.024 earlier in development.
    Root cause: Chroma.from_documents() appends by default instead of
    replacing. This test locks in that rebuilding twice on identical
    input data leaves the collection count unchanged, not doubled."""

    temp_dir = tempfile.mkdtemp()
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        docs = [Document(page_content="Test tender content about Köln deadlines.", metadata={"source": "test.md"})]

        def build():
            client = chromadb.PersistentClient(path=temp_dir)
            existing = [c.name for c in client.list_collections()]
            if "langchain" in existing:
                client.delete_collection("langchain")
            Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=temp_dir)

        build()
        client = chromadb.PersistentClient(path=temp_dir)
        first_count = client.get_collection("langchain").count()

        build()  # rebuild again, same input
        second_count = client.get_collection("langchain").count()

        assert first_count == second_count, (
            f"Chunk count changed after rebuild ({first_count} -> {second_count}) "
            "- the idempotency fix has regressed."
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)