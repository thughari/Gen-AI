import os
import sys
import time
import json
import numpy as np

from rank_bm25 import BM25Okapi

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Shared corpus
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))

sys.path.insert(0, ROOT)
# sys.path.insert(0, "../shared")
from shared.corpus import CORPUS

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

# =========================================================
# OPENAI EMBEDDINGS
# =========================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# =========================================================
# LOAD EXISTING FAISS INDEX
# =========================================================

IDX = "../lab9/faiss_insurance_index"

if os.path.exists(IDX):

    vs = FAISS.load_local(
        IDX,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("Loaded index from Lab 9 cache")

else:

    print("Lab 9 index not found. Building new index...")

    vs = FAISS.from_documents(CORPUS, embeddings)

    vs.save_local(IDX)

    print("New FAISS index built and saved")


# =========================================================
# FILTERED SEARCH FUNCTION
# =========================================================

def filtered_search(query, filters=None, k=3):

    """
    Semantic search with optional metadata filters.

    Example:
        {"policy_type": "health"}

    or:
        {"policy_type": "motor", "section": "add_ons"}
    """

    # print("in filter")
    t0 = time.time()

    # With metadata filter
    if filters:

        results = vs.similarity_search_with_score(
            query,
            k=k,
            filter=filters
        )

    # Without filter
    else:

        results = vs.similarity_search_with_score(
            query,
            k=k
        )

    latency = round((time.time() - t0) * 1000, 1)

    tag = f"FILTERED {filters}" if filters else "UNFILTERED"

    # print("\n" + "=" * 80)
    # print(f"[{tag}]")
    # print("=" * 80)

    # print(f"Query: {query}")
    # print(f"Latency: {latency} ms")
    # print(f"Results Returned: {len(results)}")

    # print("-" * 80)

    for rank, (doc, score) in enumerate(results, start=1):

        m = doc.metadata

        # print(
        #     f"{rank}. "
        #     f"[{m['doc_id']}] "
        #     f"{m['policy_type']}/{m['claim_type']} "
        #     f"score={score:.4f}"
        # )

        # print(f"   section = {m['section']}")
        # print(f"   {doc.page_content[:120]}...")
    # print(results)

    return results, latency


# =========================================================
# SMART AUTO FILTER
# =========================================================

def smart_search(query, k=3):

    """
    Auto infer policy_type from query text
    """

    q = query.lower()

    filters = None

    # Motor-related queries
    if any(word in q for word in [
        "motor",
        "car",
        "vehicle",
        "garage",
        "engine",
        "idv",
        "accident"
    ]):

        filters = {"policy_type": "motor"}

    # Life-related queries
    elif any(word in q for word in [
        "life",
        "death",
        "nominee",
        "premium waiver",
        "surrender",
        "policyholder"
    ]):

        filters = {"policy_type": "life"}

    # Health-related queries
    elif any(word in q for word in [
        "hospital",
        "surgery",
        "cashless",
        "diabetes",
        "maternity",
        "health"
    ]):

        filters = {"policy_type": "health"}

    return filtered_search(query, filters, k)

texts = [doc.page_content for doc in CORPUS]

tokenized_texts = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_texts)

def hybrid_search(query, k=3, use_filter=False):

    t0 = time.time()

    # -----------------------------------------------------
    # BM25 RESULTS
    # -----------------------------------------------------

    bm25_results = bm25_search(query, k=k * 2)

    # -----------------------------------------------------
    # FAISS RESULTS
    # -----------------------------------------------------

    if use_filter:

        policy_type = infer_policy_type(query)

        if policy_type:

            faiss_results = faiss_vs.similarity_search_with_score(
                query,
                k=k * 2,
                filter={"policy_type": policy_type}
            )

        else:

            faiss_results = faiss_vs.similarity_search_with_score(
                query,
                k=k * 2
            )

    else:

        faiss_results = faiss_vs.similarity_search_with_score(
            query,
            k=k * 2
        )

    # -----------------------------------------------------
    # RRF FUSION
    # -----------------------------------------------------

    hybrid_results = rrf_fuse(
        bm25_results,
        faiss_results
    )[:k]

    latency = round(
        (time.time() - t0) * 1000,
        1
    )

    # -----------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------

    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    print(f"Latency: {latency} ms")

    if use_filter:
        print(f"Policy Filter: {infer_policy_type(query)}")

    print("\nRANK | BM25           | FAISS          | HYBRID")
    print("-" * 65)

    for i in range(k):

        bm = (
            bm25_results[i][0].metadata["doc_id"]
            if i < len(bm25_results)
            else "-"
        )

        fa = (
            faiss_results[i][0].metadata["doc_id"]
            if i < len(faiss_results)
            else "-"
        )

        hy = (
            hybrid_results[i][0].metadata["doc_id"]
            if i < len(hybrid_results)
            else "-"
        )

        print(
            f"{i+1:<4} | "
            f"{bm:<15} | "
            f"{fa:<15} | "
            f"{hy:<15}"
        )

    return {
        "bm25": bm25_results[:k],
        "faiss": faiss_results[:k],
        "hybrid": hybrid_results,
        "latency_ms": latency
    }


def infer_policy_type(query):

    q = query.lower()

    if any(word in q for word in [
        "car",
        "motor",
        "vehicle",
        "garage",
        "engine",
        "idv",
        "accident"
    ]):
        return "motor"

    if any(word in q for word in [
        "hospital",
        "health",
        "cashless",
        "surgery",
        "maternity",
        "diabetes"
    ]):
        return "health"

    if any(word in q for word in [
        "life",
        "death",
        "nominee",
        "policyholder",
        "family",
        "pass away"
    ]):
        return "life"

    return None


def rrf_fuse(list1, list2, k=60):

    """
    Reciprocal Rank Fusion

    score += 1 / (k + rank)
    """

    scores = {}
    docs = {}

    # BM25 ranks
    for rank, (doc, _) in enumerate(list1, start=1):

        doc_id = doc.metadata["doc_id"]

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1.0 / (k + rank)
        )

        docs[doc_id] = doc

    # FAISS ranks
    for rank, (doc, _) in enumerate(list2, start=1):

        doc_id = doc.metadata["doc_id"]

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1.0 / (k + rank)
        )

        docs[doc_id] = doc

    # Sort by descending fused score
    ranked_doc_ids = sorted(
        scores.keys(),
        key=lambda x: scores[x],
        reverse=True
    )

    return [
        (docs[doc_id], scores[doc_id])
        for doc_id in ranked_doc_ids
    ]

def bm25_search(query, k=6):

    scores = bm25.get_scores(
        query.lower().split()
    )

    top_indices = np.argsort(scores)[::-1][:k]

    results = []

    for idx in top_indices:

        results.append(
            (
                CORPUS[idx],
                float(scores[idx])
            )
        )

    return results




st.set_page_config(
    page_title="InsureSafe AI Assistant",
    layout="wide"
)

with st.sidebar:
    st.title("⚙️ Settings")

    filter = st.selectbox("Filter", ["All", "health", "motor","life"])
    # temperature = st.slider("Temperature", 0.0, 1.0, 0.3)

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("💬 InsureSafe AI Assistant")
st.caption("Ask anything about insurance")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Ask your question..."):

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # smart_search
    # filtered_search
    # llm = ChatOpenAI(
    #     model="gpt",
    #     temperature=temperature,
    #     openai_api_key=os.getenv("OPENAI_API_KEY")
    # )
    result=""
    if(filter=="All"):
        result, latency = smart_search(user_input)
        print(result)
    else:
        filterRes = {"policy_type": filter}
        result, latency=filtered_search(user_input, filterRes)
        print(result)


    # print("\n" + "=" * 80)
    # print(f"[{tag}]")
    # print("=" * 80)

    # print(f"Query: {query}")
    # print(f"Latency: {latency} ms")
    # print(f"Results Returned: {len(results)}")

    # print("-" * 80)

    finalResult=""
    for rank, (doc, score) in enumerate(result, start=1):
        m = doc.metadata
        st.markdown(f"""
            ## {finalResult} rank: {rank}\n
            * doc_id: {m['doc_id']}\n
            * {m['policy_type']}/{m['claim_type']}
            * score={score:.4f}   
            * section = {m['section']}   
            * {doc.page_content}
        \n""")
        # finalResult =finalResult + f"rank: {rank}. \n"+f"[doc_id: {m['doc_id']}] \n"+f"{m['policy_type']}/{m['claim_type']} "+f"score={score:.4f}"+f"   section = {m['section']}"+f"   {doc.page_content[:120]}...\n"

    # for i in res
    # print(result)
    # lc_messages = [HumanMessage(content="You are an insurance expert.")]

    # for m in st.session_state.messages:
    #     if m["role"] == "user":
    #         lc_messages.append("Human")
    #     else:
    #         lc_messages.append("AI")

    with st.chat_message("assistant"):
        response = st.write(finalResult)

    # st.session_state.messages.append({
    #     "role": "assistant",
    #     "content": response
    # })


