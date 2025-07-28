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
from pathlib import Path

load_dotenv(override = 'TRUE')
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Book QA", layout="wide")
st.title("Choose a Book and Ask a Question")

@st.cache_resource
def load_vectordb(pdf_path: str, persist_dir: str):
    # 1) Read & clean PDF once
    reader = PdfReader(pdf_path)
    raw_pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        lines = text.splitlines()
        filtered = lines
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

# PDF selection/upload
st.sidebar.header("Book Selection")
uploads_dir = Path("uploads")
# List all PDFs in project root and uploads directory
existing_pdfs = [str(p) for p in Path(".").glob("*.pdf")]
if uploads_dir.exists():
    existing_pdfs += [str(p) for p in uploads_dir.glob("*.pdf")]
# Remove duplicates while preserving order
existing_pdfs = list(dict.fromkeys(existing_pdfs))

upload = st.sidebar.file_uploader("Upload a PDF", type="pdf")
if upload:
    uploads_dir.mkdir(exist_ok=True)
    save_path = uploads_dir / upload.name
    with open(save_path, "wb") as f:
        f.write(upload.getbuffer())
    selected_pdf = str(save_path)
    if selected_pdf not in existing_pdfs:
        existing_pdfs.append(selected_pdf)
else:
    selected_pdf = st.sidebar.selectbox("Choose a PDF", existing_pdfs)

# Use book-specific FAISS index folder
book_stem = Path(selected_pdf).stem
persist_dir = f"faiss_book_index_{book_stem}"
vectordb = load_vectordb(selected_pdf, persist_dir)
llm_client = get_llm()


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