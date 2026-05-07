import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Load env
load_dotenv()

# Page config
st.set_page_config(
    page_title="InsureSafe AI Assistant",
    layout="wide"
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("⚙️ Settings")

    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-5.4"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3)

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Chat UI ----------------
st.title("💬 InsureSafe AI Assistant")
st.caption("Ask anything about insurance")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you?"}
    ]

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Ask your question..."):

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    lc_messages = [HumanMessage(content="You are an insurance expert.")]

    for m in st.session_state.messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        response = st.write_stream(llm.stream(lc_messages))

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })