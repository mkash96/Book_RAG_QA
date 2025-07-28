from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv(override = 'TRUE')
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Book QA", layout="wide")
st.title("Ask Anything of Your Book")

@st.cache_resource
def load_vectordb(pdf_path: str, persist_dir: str):
    # 1) Read & clean PDF once
    BAD = [
        "do you feel frustrated",
        "stuck or overwhelmed",
        "free audio & video",
        "think-and-grow-rich-ebook.com"
    ]
    reader = PdfReader(pdf_path)
    raw_pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        lines = text.splitlines()
        filtered = [L for L in lines if not any(b in L.lower() for b in BAD)]
        raw_pages.append("\n".join(filtered))

    # # 2) Split per page with metadata
    # splitter = CharacterTextSplitter(
    #     separator="\n", chunk_size=1500, chunk_overlap=200, length_function=len
    # )
    # texts, metadatas = [], []
    # for i, pg in enumerate(raw_pages, start=1):
    #     for chunk in splitter.split_text(pg):
    #         texts.append(chunk)
    #         metadatas.append({"page": i})

        # 2) Treat each page as one chunk
    texts = raw_pages
    metadatas = [{"page": i} for i in range(1, len(raw_pages) + 1)]

    # 3) Build or load FAISS
    embedder = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)
    if not os.path.exists(persist_dir):
        vectordb = FAISS.from_texts(texts=texts, embedding=embedder, metadatas=metadatas)
        vectordb.save_local(persist_dir)
    else:
        vectordb = FAISS.load_local(persist_dir, embedder, allow_dangerous_deserialization=True)

    return vectordb

@st.cache_resource
def get_llm():
    return OpenAI(api_key=api_key)

# Initialize resources
PDF_PATH    = "Think-And-Grow-Rich.pdf"
PERSIST_DIR = "./faiss_book_index"
vectordb    = load_vectordb(PDF_PATH, PERSIST_DIR)
llm_client  = get_llm()

system_prompt = "You are a helpful assistant. Answer the question using ONLY the provided excerpts—quote verbatim and cite page numbers."

def answer_question(query: str, k: int = 10) -> str:
    # 1) fetch top-k chunks
    docs = vectordb.similarity_search(query, k=k)

    # 2) build excerpts block
    excerpts = []
    for i, doc in enumerate(docs, start=1):
        pg = doc.metadata.get("page", "?")
        excerpts.append(f"Excerpt {i} (p.{pg}): “{doc.page_content.strip()}”")

    user_prompt = (
        "Here are the relevant excerpts:\n\n"
        + "\n\n".join(excerpts)
        + f"\n\nQuestion: {query}\n"
        + "Please answer as instructed."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    resp = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
    )
    answer = resp.choices[0].message.content
    # collect the page numbers to return alongside the text
    pages = [doc.metadata.get("page", "?") for doc in docs]
    return answer, pages


st.header("Ask the Book 📖")
# Use a form so the app only refreshes when you hit “Ask”

with st.form(key="qa_form"):
    question = st.text_input("Your question:")
    submit  = st.form_submit_button("Ask")

if submit and question:
    answer, pages = answer_question(question)
    st.subheader("Answer")
    st.write(answer)
    st.markdown(f"**Referenced pages:** {', '.join(map(str, pages))}")