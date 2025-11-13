import requests
import streamlit as st

# -------------------------------
# FastAPI backend endpoint
# -------------------------------
API_URL = "https://qanda-assis.onrender.com/ask"

st.set_page_config(page_title="PruQandA Chat", page_icon="💬", layout="centered")

st.title("💬 PruQandA - Intelligent Chat Assistant")
st.markdown("Ask any question about the members, trips, or preferences — powered by Azure Search + OpenAI.")

# Chat state
if "history" not in st.session_state:
    st.session_state.history = []

# Input box
question = st.chat_input("Type your question...")

if question:
    # Display user message
    st.chat_message("user").write(question)

    try:
        response = requests.post(API_URL, json={"question": question})
        if response.status_code == 200:
            answer = response.json().get("answer", "No answer found.")
        else:
            answer = f"Error {response.status_code}: {response.text}"
    except Exception as e:
        answer = f"⚠️ API error: {e}"

    # Display bot message
    st.chat_message("assistant").write(answer)

    # Save history
    st.session_state.history.append((question, answer))

# Show previous conversation
if st.session_state.history:
    st.markdown("### 🕓 Chat History")
    for q, a in st.session_state.history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")
        st.markdown("---")
