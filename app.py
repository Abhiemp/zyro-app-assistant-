
import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# -----------------------------
# CONFIG
# -----------------------------
LLM_MODEL = "llama-3.3-70b-versatile"
CORPUS_PATH = "."   # keep the HR PDF folder in repo with this exact name


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Zyro HR Policy Assistant", page_icon="🤖")
st.title("🤖 Zyro HR Policy Assistant")
st.write("Ask questions about Zyro Dynamics HR policies.")

groq_api_key = st.secrets.get("GROQ_API_KEY", None)

if not groq_api_key:
    st.error("GROQ_API_KEY not found in Streamlit secrets.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key


# -----------------------------
# LOAD RAG PIPELINE ONCE
# -----------------------------
@st.cache_resource
def build_rag_pipeline():
    # Load docs
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    # Chunk docs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 40,
            "lambda_mult": 0.75
        }
    )

    # LLM
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=512
    )

    # RAG prompt
    rag_prompt = ChatPromptTemplate.from_template("""
You are an HR policy assistant for Zyro Dynamics.

Use ONLY the provided context.

Rules:
1. Answer ONLY from the context.
2. If the answer exists anywhere in the context, provide it.
3. Do not say information is missing unless the context truly lacks it.
4. Quote exact numbers, dates, limits, eligibility criteria and policy values when available.
5. Be concise but complete.
6. Never invent information.

Context:
{context}

Question:
{question}

Answer:
""")

    # OOS classifier prompt
    oos_prompt = ChatPromptTemplate.from_template("""
You are an HR query classifier for Zyro Dynamics.

Classify the question as:

YES = related to company HR policies, employee handbook, leave, payroll,
benefits, travel, reimbursement, onboarding, code of conduct, security,
performance reviews, workplace guidelines, work from home, attendance,
employee procedures.

NO = not related to company HR policies.

Return ONLY:
YES
or
NO

Examples:

Question: What is the work from home policy?
YES

Question: What are the employee code of conduct requirements?
YES

Question: What is the travel and expense reimbursement policy?
YES

Question: How does the performance review process work?
YES

Question: What are the onboarding requirements for new employees?
YES

Question: How many casual leaves are employees entitled to?
YES

Question: Who won the FIFA World Cup 2022?
NO

Question:
{question}
""")

    refusal_message = (
        "I can only answer questions related to Zyro Dynamics HR policies and employee documentation."
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_chain(question: str):
        docs = retriever.invoke(question)
        context = format_docs(docs)

        chain = rag_prompt | llm | StrOutputParser()
        return chain.invoke({
            "context": context,
            "question": question
        })

    def ask_bot(question: str):
        classifier_chain = oos_prompt | llm | StrOutputParser()
        decision = classifier_chain.invoke({"question": question}).strip().upper()

        if "NO" in decision:
            return refusal_message

        return rag_chain(question)

    return ask_bot


ask_bot = build_rag_pipeline()


# -----------------------------
# CHAT UI
# -----------------------------
# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(
    page_title="Zyro HR Policy Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    .hero-box {
        background: linear-gradient(135deg, #111827, #1f2937);
        padding: 1.5rem 1.5rem 1.2rem 1.5rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.3rem;
    }

    .hero-sub {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.5;
    }

    .chip {
        display: inline-block;
        background: #1e293b;
        color: #dbeafe;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        margin-right: 0.45rem;
        margin-top: 0.5rem;
        font-size: 0.88rem;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .user-msg {
        background: #1d4ed8;
        color: white;
        padding: 0.95rem 1rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.7rem 0 0.35rem auto;
        width: fit-content;
        max-width: 85%;
        box-shadow: 0 4px 12px rgba(0,0,0,0.18);
    }

    .bot-msg {
        background: #1f2937;
        color: #f8fafc;
        padding: 1rem 1rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.35rem 0 1rem 0;
        width: fit-content;
        max-width: 90%;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .section-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 1rem;
        margin-bottom: 0.35rem;
        font-weight: 600;
        letter-spacing: 0.2px;
    }

    .sample-box {
        background: #111827;
        border: 1px solid rgba(255,255,255,0.06);
        padding: 0.9rem 1rem;
        border-radius: 14px;
        margin-bottom: 0.75rem;
        color: #e5e7eb;
        font-size: 0.95rem;
    }

    .footer-note {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 1.2rem;
    }

    .stTextInput > div > div > input {
        border-radius: 12px;
        padding: 0.7rem 0.9rem;
    }

    .stButton button {
        border-radius: 12px;
        font-weight: 600;
        padding: 0.55rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Header ----------
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🤖 Zyro HR Policy Assistant</div>
    <div class="hero-sub">
        Ask questions about Zyro Dynamics HR policies, leave rules, work-from-home guidelines,
        travel reimbursements, onboarding, conduct, benefits, and employee procedures.
    </div>
    <div>
        <span class="chip">HR Policies</span>
        <span class="chip">Leave & Attendance</span>
        <span class="chip">WFH Rules</span>
        <span class="chip">Travel & Reimbursements</span>
        <span class="chip">Employee Handbook</span>
    </div>
</div>
""", unsafe_allow_html=True)

groq_api_key = st.secrets.get("GROQ_API_KEY", None)

if not groq_api_key:
    st.error("GROQ_API_KEY not found in Streamlit secrets.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key


# -----------------------------
# LOAD RAG PIPELINE ONCE
# -----------------------------
@st.cache_resource
def build_rag_pipeline():
    # Load docs
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    # Chunk docs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 40,
            "lambda_mult": 0.75
        }
    )

    # LLM
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=512
    )

    # RAG prompt
    rag_prompt = ChatPromptTemplate.from_template("""
You are an HR policy assistant for Zyro Dynamics.

Use ONLY the provided context.

Rules:
1. Answer ONLY from the context.
2. If the answer exists anywhere in the context, provide it.
3. Do not say information is missing unless the context truly lacks it.
4. Quote exact numbers, dates, limits, eligibility criteria and policy values when available.
5. Be concise but complete.
6. Never invent information.

Context:
{context}

Question:
{question}

Answer:
""")

    # OOS classifier prompt
    oos_prompt = ChatPromptTemplate.from_template("""
You are an HR query classifier for Zyro Dynamics.

Classify the question as:

YES = related to company HR policies, employee handbook, leave, payroll,
benefits, travel, reimbursement, onboarding, code of conduct, security,
performance reviews, workplace guidelines, work from home, attendance,
employee procedures.

NO = not related to company HR policies.

Return ONLY:
YES
or
NO

Examples:

Question: What is the work from home policy?
YES

Question: What are the employee code of conduct requirements?
YES

Question: What is the travel and expense reimbursement policy?
YES

Question: How does the performance review process work?
YES

Question: What are the onboarding requirements for new employees?
YES

Question: How many casual leaves are employees entitled to?
YES

Question: Who won the FIFA World Cup 2022?
NO

Question:
{question}
""")

    refusal_message = (
        "I can only answer questions related to Zyro Dynamics HR policies and employee documentation."
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_chain(question: str):
        docs = retriever.invoke(question)
        context = format_docs(docs)

        chain = rag_prompt | llm | StrOutputParser()
        return chain.invoke({
            "context": context,
            "question": question
        })

    def ask_bot(question: str):
        classifier_chain = oos_prompt | llm | StrOutputParser()
        decision = classifier_chain.invoke({"question": question}).strip().upper()

        if "NO" in decision:
            return refusal_message

        return rag_chain(question)

    return ask_bot


ask_bot = build_rag_pipeline()


# -----------------------------
# SESSION STATE
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "question_input" not in st.session_state:
    st.session_state.question_input = ""


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("📘 Quick Guide")
    st.write("Try asking questions like:")

    sample_questions = [
        "How many casual leaves are employees entitled to?",
        "What is the work from home eligibility criteria?",
        "What is the travel reimbursement process?",
        "What is the probation period for new employees?",
        "What are the code of conduct rules for employees?",
        "What benefits are available to employees?"
    ]

    for q in sample_questions:
        st.markdown(f"""
        <div class="sample-box">{q}</div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑 Clear chat history", use_container_width=True):
        st.session_state.history = []
        st.session_state.question_input = ""
        st.rerun()

    st.markdown(
        "<div class='footer-note'>Built for Zyro Dynamics HR document question answering.</div>",
        unsafe_allow_html=True
    )


# -----------------------------
# MAIN INPUT AREA
# -----------------------------
st.markdown("<div class='section-label'>Ask a question</div>", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])

with col1:
    question = st.text_input(
        "",
        placeholder="e.g. What is the hotel allowance for L7 to L8 during international travel?",
        key="question_input",
        label_visibility="collapsed"
    )

with col2:
    ask_clicked = st.button("Ask", use_container_width=True)


# -----------------------------
# ASK LOGIC
# -----------------------------
if ask_clicked and question.strip():
    with st.spinner("Searching policies and generating answer..."):
        try:
            answer = ask_bot(question.strip())
            st.session_state.history.append({
                "question": question.strip(),
                "answer": answer
            })
            st.session_state.question_input = ""
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")


# -----------------------------
# CHAT HISTORY
# -----------------------------
# -----------------------------
# CHAT UI
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

def submit_question():
    q = st.session_state.question_input.strip()
    if not q:
        return

    try:
        answer = ask_bot(q)
        st.session_state.history.append({
            "question": q,
            "answer": answer
        })
        st.session_state.question_input = ""
    except Exception as e:
        st.session_state.history.append({
            "question": q,
            "answer": f"Error: {e}"
        })

st.markdown("### Ask a question")

col1, col2 = st.columns([5, 1])

with col1:
    st.text_input(
        "Ask a question",
        key="question_input",
        placeholder="e.g. What is the leave policy for probation employees?",
        label_visibility="collapsed"
    )

with col2:
    st.button("Ask", use_container_width=True, on_click=submit_question)

# conversation display
if st.session_state.history:
    st.markdown("### Conversation")

    for chat in reversed(st.session_state.history):
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1d4ed8, #2563eb);
                padding: 16px 18px;
                border-radius: 16px;
                margin: 14px 0 10px 140px;
                color: white;
                box-shadow: 0 6px 18px rgba(37,99,235,0.25);
            ">
                <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 6px;"><b>You:</b></div>
                <div style="font-size: 1rem;">{chat['question']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.08);
                padding: 18px;
                border-radius: 16px;
                margin: 0 140px 18px 0;
                box-shadow: 0 6px 18px rgba(0,0,0,0.18);
            ">
                <div style="font-size: 0.95rem; color: #cbd5e1; margin-bottom: 8px;"><b>Zyro HR Assistant:</b></div>
                <div style="font-size: 1rem; line-height: 1.7;">{chat['answer']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
