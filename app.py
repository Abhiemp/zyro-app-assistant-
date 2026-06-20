
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
CORPUS_PATH = "zyro-dynamics-hr-corpus"   # keep the HR PDF folder in repo with this exact name


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
if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Enter your question:")

if st.button("Ask") and question.strip():
    try:
        answer = ask_bot(question.strip())
        st.session_state.history.append({
            "question": question.strip(),
            "answer": answer
        })
    except Exception as e:
        st.error(f"Error: {e}")

if st.session_state.history:
    st.subheader("Conversation History")
    for chat in reversed(st.session_state.history):
        st.markdown(f"**Question:** {chat['question']}")
        st.markdown(f"**Answer:** {chat['answer']}")
        st.markdown("---")
