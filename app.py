
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
# =========================================================
# UI SECTION
if "history" not in st.session_state:
    st.session_state.history = []

if "question_input" not in st.session_state:
    st.session_state.question_input = ""


# -----------------------------
# SUBMIT HANDLER
# -----------------------------
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


# -----------------------------
# PAGE STYLING
# -----------------------------
st.markdown("""
<style>
/* =========================================================
   GLOBAL APP
========================================================= */
.stApp {
    background:
        linear-gradient(rgba(4, 10, 24, 0.42), rgba(4, 10, 24, 0.68)),
        url("https://cdn.dribbble.com/userupload/6508147/file/original-11794da09cf2fda821918b0be52877cf.jpg");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: #f8fafc;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

section[data-testid="stSidebar"] {
    background: rgba(7, 12, 24, 0.7);
}

/* Hide streamlit default top spacing feel */
[data-testid="stDecoration"] {
    display: none;
}

/* =========================================================
   WRAPPER
========================================================= */
.main-shell {
    position: relative;
    z-index: 1;
}

/* =========================================================
   HERO CARD
========================================================= */
.hero-card {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    padding: 34px 34px 28px 34px;
    margin-bottom: 28px;
    background:
        linear-gradient(135deg, rgba(8, 15, 32, 0.82), rgba(17, 25, 45, 0.70));
    border: 1px solid rgba(255,255,255,0.10);
    backdrop-filter: blur(12px);
    box-shadow:
        0 16px 48px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.06);
}

.hero-card {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    padding: 34px 34px 28px 34px;
    margin-bottom: 28px;
    background:
        linear-gradient(135deg, rgba(7, 12, 28, 0.78), rgba(14, 22, 40, 0.68));
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(14px);
    box-shadow:
        0 16px 48px rgba(0,0,0,0.38),
        inset 0 1px 0 rgba(255,255,255,0.06);
}

.hero-title {
    position: relative;
    z-index: 1;
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.1;
    color: #ffffff;
    margin-bottom: 14px;
    letter-spacing: -0.02em;
}

.hero-sub {
    position: relative;
    z-index: 1;
    max-width: 920px;
    font-size: 1.06rem;
    line-height: 1.75;
    color: #d7e5f3;
}

.hero-mini {
    position: relative;
    z-index: 1;
    margin-top: 20px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    border-radius: 999px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.10);
    color: #eef6ff;
    font-size: 0.95rem;
    backdrop-filter: blur(6px);
}

/* =========================================================
   SECTION TITLE
========================================================= */
.section-title {
    font-size: 1.12rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 8px 0 12px 2px;
}

/* =========================================================
   ASK BOX
========================================================= */
.ask-shell {
    border-radius: 24px;
    padding: 22px;
    margin-bottom: 24px;
    background: rgba(7, 12, 28, 0.76);
    border: 1px solid rgba(255,255,255,0.10);
    backdrop-filter: blur(12px);
    box-shadow: 0 14px 36px rgba(0,0,0,0.30);
}
div[data-baseweb="input"] > div {
    background: rgba(5, 11, 24, 0.88) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 16px !important;
    min-height: 56px !important;
}

div[data-baseweb="input"] input {
    color: #f8fafc !important;
    font-size: 1rem !important;
}

div[data-baseweb="input"] input::placeholder {
    color: #94a3b8 !important;
}

.stButton > button {
    height: 56px;
    width: 100%;
    border-radius: 16px !important;
    border: 1px solid rgba(96,165,250,0.35) !important;
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    box-shadow: 0 10px 24px rgba(37,99,235,0.30);
    transition: 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 30px rgba(37,99,235,0.38);
}

/* =========================================================
   CHIPS
========================================================= */
.chips-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 10px;
}

.chip {
    padding: 9px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    color: #dbeafe;
    font-size: 0.92rem;
}

/* =========================================================
   CHAT AREA
========================================================= */
.chat-wrap {
    margin-top: 12px;
}

.user-bubble {
    margin: 14px 0 10px auto;
    max-width: 78%;
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white;
    padding: 16px 18px;
    border-radius: 20px 20px 6px 20px;
    box-shadow: 0 8px 24px rgba(37,99,235,0.28);
    border: 1px solid rgba(255,255,255,0.08);
}

.bot-bubble {
    margin: 0 auto 18px 0;
    max-width: 82%;
    background: rgba(8, 14, 28, 0.76);
    color: #f8fafc;
    padding: 18px;
    border-radius: 20px 20px 20px 6px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    backdrop-filter: blur(8px);
}

.bubble-head {
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 8px;
    opacity: 0.92;
}

.bubble-body {
    font-size: 1rem;
    line-height: 1.8;
    white-space: pre-wrap;
}

/* =========================================================
   FOOTER
========================================================= */
.footer-note {
    margin-top: 26px;
    color: #cbd5e1;
    font-size: 0.92rem;
    text-align: center;
    opacity: 0.95;
}

/* =========================================================
   MOBILE
========================================================= */
@media (max-width: 900px) {
    .hero-title { font-size: 1.9rem; }
    .hero-sub { font-size: 0.98rem; }
    .user-bubble, .bot-bubble { max-width: 100%; }
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# MAIN WRAPPER
# -----------------------------
st.markdown('<div class="main-shell">', unsafe_allow_html=True)


# -----------------------------
# HERO
# -----------------------------
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🤖 Zyro HR Policy Assistant</div>
    <div class="hero-sub">
        Ask questions about <b>Zyro Dynamics</b> HR policies, leave rules, work-from-home guidelines,
        travel reimbursements, onboarding, code of conduct, benefits, employee handbook rules,
        and internal workplace procedures.
    </div>

  
</div>
""", unsafe_allow_html=True)


# -----------------------------
# ASK BOX
# -----------------------------
st.markdown('<div class="section-title">Ask a question</div>', unsafe_allow_html=True)
col1, col2 = st.columns([6, 1.25])

with col1:
    st.text_input(
        "Ask a question",
        key="question_input",
        placeholder="e.g. What is the probation period for new employees?",
        label_visibility="collapsed"
    )

with col2:
    st.button("Ask", use_container_width=True, on_click=submit_question)

st.markdown("""
<div class="chips-row">
    <span class="chip">Leave policy</span>
    <span class="chip">Probation period</span>
    <span class="chip">Travel claims</span>
    <span class="chip">WFH eligibility</span>
    <span class="chip">Code of conduct</span>
    <span class="chip">Performance review</span>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# CONVERSATION
# -----------------------------
if st.session_state.history:
    st.markdown('<div class="section-title">Conversation</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

    for chat in reversed(st.session_state.history):
        question_text = (
            str(chat["question"])
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        answer_text = (
            str(chat["answer"])
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\\n", "<br>")
            .replace("\n", "<br>")
        )

        st.markdown(
            f"""
            <div class="user-bubble">
                <div class="bubble-head">You</div>
                <div class="bubble-body">{question_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="bot-bubble">
                <div class="bubble-head">Zyro HR Assistant</div>
                <div class="bubble-body">{answer_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<div class="footer-note">
    Built for the Zyro Dynamics HR RAG challenge • Answers are restricted to the uploaded company policy corpus
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
