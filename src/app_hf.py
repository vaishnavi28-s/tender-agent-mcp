import sys
import os
import gradio as gr

sys.path.append(os.path.dirname(__file__))
from llm import answer_query


def ask_tenderbot(message: str, history: list) -> str:
    print("HISTORY FORMAT:", history)
    if not message or not message.strip():
        return "Bitte gib eine Frage ein."
    return answer_query(message, history=history)


demo = gr.ChatInterface(
    fn=ask_tenderbot,
    title="TenderBot",
    description=(
        "Fragen zu deutschen Ausschreibungen für Wahlunterlagen (service.bund.de). "
        "Antworten basieren ausschließlich auf indexierten Ausschreibungsdaten."
    ),
    examples=[
        "Zeige mir alle aktuellen Ausschreibungen zur Wahlunterlagen-Erstellung.",
        "Was sind die Anforderungen für Köln?",
        "Bis wann müssen die Wahlunterlagen zugestellt werden?",
    ],
)

if __name__ == "__main__":
    demo.launch(mcp_server=True)