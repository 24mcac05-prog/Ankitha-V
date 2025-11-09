jarvis-aI
Jarvis is a lightweight, local-first Streamlit document assistant that lets you upload a single PDF/DOCX/TXT file and ask questions that are answered only from the uploaded document. It uses a fast TF-IDF + cosine similarity retriever by default and offers optional integrations for Pinecone (vector index + sentence-transformer embeddings) and Ollama (local LLM such as tinyllama) for nicer summarization — however retrieval answers always come from your documents. The app includes a simple SQLite-backed email/password signup & login (PBKDF2-hashed passwords) for local/dev use, a five-line summarizer, voice-record and TTS helpers (optional), chat history, and export-to-PDF/DOCX features. To run locally: create a Python virtual environment, install dependencies (streamlit, PyPDF2, python-docx, fpdf, optional sentence-transformers and pinecone-client/ollama), set PINECONE_API_KEY and PINECONE_INDEX in a .env if you want Pinecone, optionally set OLLAMA_MODEL=tinyllama to use Ollama, then start with streamlit run app_with_pinecone_ollama.py. Security note: the built-in auth is intended for local/dev only — do not use it as-is for production.
jarvis/
├─ app_with_pinecone_ollama.py        # Main Streamlit app (Pinecone + Ollama )
├─ requirements.txt                   # Python deps (streamlit, pypdf2, python-docx, fpdf, etc.)
├─ .env                               # API keys & config (not committed)
├─ users.db                           # SQLite file for local auth (auto-created)
├─ uploads/                           # (optional) persisted uploaded files
├─ data/
│  ├─ indexes/                        # (optional) local index files or embeddings cache
│  └─ logs/                           # runtime logs (optional)
├─ scripts/
│  ├─ start_streamlit.ps1             # convenience script to run on Windows
│  └─ start_streamlit.sh              # convenience script to run on Mac/Linux
├─ README.md
└─ tests/
   └─ test_retriever.py
vector search / embeddings
Pinecone (managed vector DB) — optional remote index for storing/retrieving document embeddings at scale.
Usage: encode chunks with a sentence-embedding model → pinecone.Index.upsert() → query() to fetch nearest chunks.
Config via .env: PINECONE_API_KEY, PINECONE_INDEX.
Notes: modern Pinecone client uses Pinecone(api_key=...) class. App falls back to TF-IDF when Pinecone not configured or unavailable.

sentence-transformers (e.g. all-MiniLM-L6-v2) — for creating embeddings to upsert/query Pinecone.
local LLM
Ollama (e.g. tinyllama, phi3) — local LLM used only for nicer summarization or paraphrasing; the app keeps retrieval answers strictly from uploaded docs.
Usage: run ollama serve locally and set OLLAMA_MODEL in .env (example: OLLAMA_MODEL=tinyllama).
App uses langchain_community.llms.Ollama if available; falls back gracefully.
Auth & persistence

