import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

BOT_NAME = "rizvi pdf AI"

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title=BOT_NAME, page_icon="🤖")
st.title("🤖 " + BOT_NAME)
st.caption("Upload your pdf here and ask your questions😊")


@st.cache_resource
def load_llm():
    return ChatGroq(model="openai/gpt-oss-20b")


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def build_db(uploaded_file, _embeddings):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    pages = PyPDFLoader(tmp_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    os.remove(tmp_path)
    return FAISS.from_documents(docs, _embeddings)


def ask(db, llm, question):
    found = db.similarity_search(question, k=4)
    context = "\n".join(d.page_content for d in found)
    prompt = f"""You are a friendly assistant for a business.
Answer ONLY using the context below.
Reply in the same language and style the user wrote in
(English, Hindi, Hinglish, Urdu, etc.).
Keep the answer short and simple.
If the answer is not in the context, politely say you don't have this
information.

Context:
{context}

Question: {question}"""
    return llm.invoke(prompt).content


llm = load_llm()
embeddings = load_embeddings()

uploaded_file = st.file_uploader("Upload your PDF here 👇", type="pdf")

if uploaded_file is None:
    st.info("Upload Your file to start chatting.")
    st.stop()

if "db" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
    with st.spinner("PDF padh raha hoon..."):
        st.session_state.db = build_db(uploaded_file, embeddings)
        st.session_state.file_name = uploaded_file.name
        st.session_state.messages = []

st.success(f"'{uploaded_file.name}' ready hai — ab sawaal poochho!")

for m in st.session_state.get("messages", []):
    with st.chat_message(m["role"]):
        st.write(m["content"])

q = st.chat_input("Ask Your Questions here")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            a = ask(st.session_state.db, llm, q)
        st.write(a)
    st.session_state.messages.append({"role": "assistant", "content": a})
