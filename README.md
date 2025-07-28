# Book QA (Book_RAG)

A Streamlit-based Retrieval-Augmented Generation (RAG) app that lets you upload or select a PDF “book,” choose between multiple LLM backends (OpenAI, Anthropic, Ollama), and ask questions—quoting verbatim from the text and citing page numbers.

## Demo & Screenshots
See [docs/Demo.pdf](docs/Demo.pdf) for a UI walkthrough and model-comparison.

## Features

- **PDF management**  
  - Upload new PDFs or pick from existing ones in your project root or `uploads/`.
  - Each PDF gets its own FAISS index folder (`faiss_book_index_<stem>`).
- **Multi-model support**  
  - Select from `gpt-4o-mini` (OpenAI), `claude-3-haiku-20240307` (Anthropic), `llama3.2` or `mistral:7b` (via Ollama).
- **RAG pipeline**  
  - Breaks PDF into page-level chunks, embeds with OpenAI’s `text-embedding-ada-002`, and retrieves top-k.
  - Builds system + user prompts to answer questions using only the excerpts.

## Setup

1. **Clone the repo**  
   ```bash
   git clone <your-repo-url>
   cd Book_RAG
   ```

2. **Create & activate environment**  
    ```bash   
    conda env create -f environment.yaml
    conda activate book-rag
    #or with pip
    pip install -r requirements.txt
    ```
    
3. **Configure secrets**
    Create a .env in the project root: 
    
    OPENAI_API_KEY=your_openai_key
    ANTHROPIC_API_KEY=your_anthropic_key

    If you’re using an Ollama server, update its base URL in app.py.

4. **Run the app**
    ```bash
    streamlit run app.py
    ```

## Usage

1.    In the sidebar, upload a new PDF or select an existing one.
2.    Pick your model from the “Model Selection” dropdown.
3.    Type your question and hit Ask.
4.    View the answer and referenced pages.

