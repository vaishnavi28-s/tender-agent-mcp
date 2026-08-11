import sys
import os
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__)))
from llm import answer_query

st.set_page_config(page_title="TenderBot", layout="wide")
st.title("TenderBot - Ausschreibungsfragen stellen")

if "chat" not in st.session_state:
    st.session_state.chat = []

st.markdown("#### Beispiel-Fragen:")
st.markdown("- Zeige mir alle aktuellen Ausschreibungen zur Wahlunterlagen-Erstellung.")
st.markdown("- Welche Leistungen umfasst der Auftrag der Stadt Köln zur Scandienstleistung für die Kommunalwahl 2025?")

user_input = st.chat_input("Frage eingeben (z. B. Welche Unterlagen muss ich in Münster einreichen?)")
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
    for speaker, message in st.session_state.chat:
        if speaker.startswith("You"):
            st.markdown(f"**{speaker}:** {message}")
        else:
            st.markdown(f"**{speaker}:**")
            st.markdown(message)