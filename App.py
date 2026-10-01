import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

PDF_PATH = "digital_marketing_rag_practical-3.pdf"
BOT_NAME = "Digital Marketing Assistant"

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title=BOT_NAME, page_icon="🤖")
st.title("🤖 " + BOT_NAME)
st.caption("Hinglish, English ya kisi bhi language mein poochho")


@st.cache_resource
def load_db():
    pages = PyPDFLoader(PDF_PATH).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return FAISS.from_documents(docs, emb)


@st.cache_resource
def load_llm():
    return ChatGroq(model="openai/gpt-oss-20b")


db = load_db()
llm = load_llm()


def ask(question):
    found = db.similarity_search(question, k=4)
    context = "\n".join(d.page_content for d in found)
    prompt = f"""You are a friendly assistant for a business.
Answer ONLY using the context below.
Reply in the same language and style the user wrote in
(English, Hindi, Hinglish, Urdu, etc.).
Keep the answer short and simple.
If the answer is not in the context, politely say you don't have this
information and ask them to contact the team.

Context:
{context}

Question: {question}"""
    return llm.invoke(prompt).content


if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

q = st.chat_input("Apna sawaal likho...")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            a = ask(q)
        st.write(a)
    st.session_state.messages.append({"role": "assistant", "content": a})
