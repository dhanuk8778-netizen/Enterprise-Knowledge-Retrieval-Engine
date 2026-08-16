"""
Streamlit UI for the Enterprise Agentic RAG Platform.

A thin client over the FastAPI backend — it does no retrieval, chunking, or
generation itself, it just calls the API. This keeps the UI swappable
(React/Next.js later) without touching the RAG pipeline.

Run:
    streamlit run streamlit_app.py

Configure the backend URL via the API_BASE_URL env var (defaults to
http://localhost:8000, or http://api:8000 automatically inside Docker).
"""
import os
import time

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Agentic RAG Platform", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "token" not in st.session_state:
    st.session_state.token = None
if "email" not in st.session_state:
    st.session_state.email = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: role, content, citations, route, latency_ms


def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


def api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Sidebar: connection status + auth
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 Agentic RAG")
    st.caption(f"Backend: `{API_BASE_URL}`")

    if api_health():
        st.success("API reachable")
    else:
        st.error("API unreachable — start the backend first (`uvicorn app.main:app`).")

    st.divider()

    if not st.session_state.token:
        st.subheader("Sign in")
        tab_login, tab_register = st.tabs(["Login", "Register"])

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Log in", use_container_width=True)
            if submitted:
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/auth/login",
                        data={"username": email, "password": password},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        st.session_state.token = resp.json()["access_token"]
                        st.session_state.email = email
                        st.rerun()
                    else:
                        st.error(resp.json().get("detail", "Login failed"))
                except requests.RequestException as e:
                    st.error(f"Could not reach API: {e}")

        with tab_register:
            with st.form("register_form"):
                email_r = st.text_input("Email", key="reg_email")
                password_r = st.text_input("Password (min 8 chars)", type="password", key="reg_password")
                submitted_r = st.form_submit_button("Create account", use_container_width=True)
            if submitted_r:
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/auth/register",
                        json={"email": email_r, "password": password_r},
                        timeout=10,
                    )
                    if resp.status_code == 201:
                        st.session_state.token = resp.json()["access_token"]
                        st.session_state.email = email_r
                        st.rerun()
                    else:
                        st.error(resp.json().get("detail", "Registration failed"))
                except requests.RequestException as e:
                    st.error(f"Could not reach API: {e}")
    else:
        st.success(f"Signed in as **{st.session_state.email}**")
        if st.button("Log out", use_container_width=True):
            st.session_state.token = None
            st.session_state.email = None
            st.session_state.conversation_id = None
            st.session_state.history = []
            st.rerun()

        st.divider()
        st.subheader("📄 Documents")

        uploaded = st.file_uploader(
            "Upload PDF / DOCX / TXT / CSV",
            type=["pdf", "docx", "txt", "md", "csv"],
        )
        if uploaded is not None and st.button("Ingest document", use_container_width=True):
            with st.spinner(f"Ingesting {uploaded.name}…"):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue())}
                    resp = requests.post(
                        f"{API_BASE_URL}/documents/upload",
                        headers=auth_headers(),
                        files=files,
                        timeout=60,
                    )
                    if resp.status_code == 201:
                        doc = resp.json()
                        st.success(f"Ingested '{doc['filename']}' ({doc['status']})")
                    else:
                        st.error(resp.json().get("detail", "Upload failed"))
                except requests.RequestException as e:
                    st.error(f"Could not reach API: {e}")

        try:
            docs_resp = requests.get(f"{API_BASE_URL}/documents", headers=auth_headers(), timeout=10)
            if docs_resp.status_code == 200:
                docs = docs_resp.json()
                if docs:
                    st.caption(f"{len(docs)} document(s) indexed")
                    for d in docs:
                        col1, col2 = st.columns([4, 1])
                        col1.write(f"📄 {d['filename']} · `{d['status']}`")
                        if col2.button("🗑️", key=f"del_{d['id']}"):
                            requests.delete(
                                f"{API_BASE_URL}/documents/{d['id']}", headers=auth_headers(), timeout=10
                            )
                            st.rerun()
                else:
                    st.caption("No documents yet — upload one above.")
        except requests.RequestException:
            pass

        st.divider()
        if st.button("🔄 New conversation", use_container_width=True):
            st.session_state.conversation_id = None
            st.session_state.history = []
            st.rerun()

# ---------------------------------------------------------------------------
# Main panel: chat / Q&A
# ---------------------------------------------------------------------------
st.header("Ask your documents")
st.caption(
    "Answers are grounded in retrieved chunks and cite their source. "
    "Routing, retrieval, and reranking happen server-side — this UI just displays the result."
)

if not st.session_state.token:
    st.info("👈 Sign in or create an account in the sidebar to get started.")
    st.stop()

for turn in st.session_state.history:
    with st.chat_message("user" if turn["role"] == "user" else "assistant"):
        st.write(turn["content"])
        if turn["role"] == "assistant":
            meta_cols = st.columns(3)
            meta_cols[0].caption(f"route: `{turn.get('route', '-')}`")
            meta_cols[1].caption(f"latency: {turn.get('latency_ms', 0):.0f} ms")
            meta_cols[2].caption(f"{len(turn.get('citations', []))} citation(s)")
            for c in turn.get("citations", []):
                with st.expander(f"[{c['marker']}] source chunk `{c['chunk_id'][:8]}…`"):
                    st.write(c["snippet"])

query = st.chat_input("Ask a question about your uploaded documents…")
if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Routing → retrieving → reranking → generating…"):
            try:
                t0 = time.perf_counter()
                resp = requests.post(
                    f"{API_BASE_URL}/query",
                    headers=auth_headers(),
                    json={"query": query, "conversation_id": st.session_state.conversation_id},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.conversation_id = data["conversation_id"]
                    st.write(data["answer"])

                    meta_cols = st.columns(3)
                    meta_cols[0].caption(f"route: `{data['route']}`")
                    meta_cols[1].caption(f"latency: {data['latency_ms']:.0f} ms")
                    meta_cols[2].caption(f"{len(data['citations'])} citation(s)")

                    for c in data["citations"]:
                        with st.expander(f"[{c['marker']}] source chunk `{c['chunk_id'][:8]}…`"):
                            st.write(c["snippet"])

                    st.session_state.history.append(
                        {
                            "role": "assistant",
                            "content": data["answer"],
                            "citations": data["citations"],
                            "route": data["route"],
                            "latency_ms": data["latency_ms"],
                        }
                    )
                else:
                    st.error(resp.json().get("detail", "Query failed"))
            except requests.RequestException as e:
                st.error(f"Could not reach API: {e}")
