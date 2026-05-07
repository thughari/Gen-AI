# =========================================================
# InsureSafe Pro — Full‑Stack Insurance Chatbot
# Streamlit + LangChain + Docling + Memory
# =========================================================

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from docling.document_converter import DocumentConverter

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="InsureSafe Pro",
    page_icon="■",
    layout="wide"
)

# ---------------------------------------------------------
# Session State
# ---------------------------------------------------------
def init_session():
    defaults = {
        "messages": [],
        "policy_context": "",
        "policy_name": None,
        "policy_summary": None,
        "doc_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
SAMPLE_PDF_PATH = "./lab3/sample_pdfs/health_policy.pdf"

# ---------------------------------------------------------
# Docling PDF Extraction
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def extract_policy_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using Docling."""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()

@st.cache_data(show_spinner=False)
def extract_policy_text_from_upload(pdf_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        path = tmp.name
    try:
        converter = DocumentConverter()
        result = converter.convert(path)
        return result.document.export_to_markdown()
    finally:
        os.unlink(path)

# ---------------------------------------------------------
# Policy Summary
# ---------------------------------------------------------
def generate_policy_summary(text: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = HumanMessage(
        content=(
            "Summarize the insurance policy below.\n\n"
            "Create a markdown table with the following fields:\n"
            "- Coverage\n"
            "- Exclusions\n"
            "- Premiums\n"
            "- Waiting Periods\n\n"
            f"POLICY DOCUMENT:\n{text[:6000]}"
        )
    )

    return llm.invoke([prompt]).content

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("■ Policy Source")

    uploaded = st.file_uploader(
        "Upload Policy PDF",
        type=["pdf"]
    )

    if uploaded:
        with st.spinner("■ Analyzing uploaded document..."):
            text = extract_policy_text_from_upload(uploaded.read())
            st.session_state.policy_context = text
            st.session_state.policy_name = uploaded.name
            st.session_state.doc_loaded = True
            st.session_state.messages = []

        with st.spinner("■ Generating policy summary..."):
            st.session_state.policy_summary = generate_policy_summary(text)

        st.success(f"Loaded: {uploaded.name}")

    if st.button("▶ Use Sample Policy (files/sample.pdf)"):
        if not os.path.exists(SAMPLE_PDF_PATH):
            st.error("Sample PDF not found at files/sample.pdf")
        else:
            with st.spinner("■ Analyzing sample policy..."):
                text = extract_policy_text_from_pdf(SAMPLE_PDF_PATH)
                st.session_state.policy_context = text
                st.session_state.policy_name = "sample.pdf"
                st.session_state.doc_loaded = True
                st.session_state.messages = []

            with st.spinner("■ Generating policy summary..."):
                st.session_state.policy_summary = generate_policy_summary(text)

            st.success("Loaded sample policy")

    st.divider()

    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3)

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_chat, tab_summary = st.tabs(["💬 Chat", "📄 Policy Summary"])

# =========================================================
# CHAT TAB — PERMANENTLY FIXED
# =========================================================
with tab_chat:
    col1, col2 = st.columns([3, 1])

    # ---------------- Chat Column ----------------
    with col1:
        st.title("■ InsureSafe Pro Advisor")

        system_msg = SystemMessage(
            content=(
                "You are an expert insurance advisor.\n"
                "Use ONLY the policy document provided below.\n"
                "If something is not mentioned in the document, say so clearly.\n\n"
                f"{st.session_state.policy_context[:6000]}"
                if st.session_state.doc_loaded
                else "You are an expert insurance advisor. Answer general insurance questions."
            )
        )

        st.caption(
            f"■ Grounded on: {st.session_state.policy_name}"
            if st.session_state.doc_loaded
            else "■ General knowledge mode"
        )

        # ✅ Scrollable message area (KEY FIX)
        with st.container(height=450):
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # ✅ Chat input OUTSIDE scroll container
        user_input = st.chat_input("Ask about this policy...")

        if user_input:
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )

            lc_msgs = [system_msg] + [
                HumanMessage(m["content"]) if m["role"] == "user"
                else AIMessage(m["content"])
                for m in st.session_state.messages
            ]

            llm = ChatOpenAI(model=model, temperature=temperature)

            with st.chat_message("assistant"):
                response = st.write_stream(llm.stream(lc_msgs))

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

    # ---------------- Policy Preview ----------------
    with col2:
        st.subheader("■ Policy Preview")

        if st.session_state.doc_loaded:
            st.markdown(
                f"""
                <div style="max-height:400px;
                            overflow:auto;
                            border:1px solid #333;
                            padding:10px;">
                {st.session_state.policy_context[:1200]}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Upload a PDF or use the sample policy")

# =========================================================
# SUMMARY TAB
# =========================================================
with tab_summary:
    st.subheader("■ Policy Summary")

    if st.session_state.doc_loaded:
        st.markdown(st.session_state.policy_summary)
    else:
        st.info("No policy loaded yet")