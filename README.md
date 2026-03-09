# 🦜 AI Content Analyst — Summarize & Chat with Any URL

> Load YouTube videos and web articles, get AI-powered summaries, and **chat across multiple sources at once** using Retrieval-Augmented Generation (RAG).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Core-green)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🔗 Live Demo

👉 **[your-app.streamlit.app](https://your-app.streamlit.app)** ← replace after deploying

---

## ✨ Features

- **Multi-URL loading** — paste several YouTube links or web articles and process them all
- **Cross-source RAG chat** — ask questions that are answered from all loaded sources simultaneously using a dynamically merged FAISS vector index
- **Conversation memory** — the chat remembers prior turns so you can say *"tell me more about that"* and get a coherent follow-up
- **Map-reduce summarization** — long content is chunked, each chunk summarized independently, then combined into one coherent final summary
- **URL history with caching** — previously indexed URLs are stored in session; switch between them without re-processing
- **Download summaries** — export as `.txt`, `.pdf`, or `.docx`
- **Source transparency** — every answer shows the exact passages it was drawn from

---

## 🏗️ Architecture

```
URL(s) Input
    │
    ├─► YouTube  ──► youtube-transcript-api  ─┐
    └─► Web page ──► UnstructuredURLLoader   ─┤
                                              ▼
                              RecursiveCharacterTextSplitter
                                              │
                              ┌───────────────┴──────────────┐
                              ▼                              ▼
                     Map-Reduce Summary            FAISS Vector Index
                     (LLM per chunk +              (HuggingFace Embeddings)
                      combine step)                          │
                              │                              ▼
                         Summary Card               RAG Chat (with memory)
                       + Download (TXT/PDF/DOCX)    Top-K retrieval → LLM answer
```

Multiple URLs → individual FAISS indexes → `merge_from()` → unified index → single query answers across all sources.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq API — `llama-3.1-8b-instant` |
| Embeddings | HuggingFace — `all-MiniLM-L6-v2` (local, free) |
| Vector store | FAISS (in-memory) |
| Orchestration | LangChain Core |
| YouTube loader | `youtube-transcript-api` v1.x |
| Web loader | `UnstructuredURLLoader` |
| PDF export | `fpdf2` |
| DOCX export | `python-docx` |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/muneebaumarr/AI-Content-Analyst.git
cd ai-content-analyst
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your Groq API key

Get a free key at [console.groq.com](https://console.groq.com), then either:

- Enter it in the app sidebar at runtime (easiest), or
- Create a `.env` file:

```env
GROQ_API_KEY=gsk_your_key_here
```

### 5. Run the app

```bash
streamlit run summarizer_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📖 Usage

1. Enter your **Groq API key** in the sidebar
2. Paste a **YouTube or web URL** and click **Load & Index**
3. Repeat for as many URLs as you want — each is cached automatically
4. Read the **summary tab** for each source
5. Use the **chat panel** to ask questions — the model searches across all active sources and remembers the conversation

---

## 🌐 Deploy on Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select this repo → set main file to `app.py`
4. Add `GROQ_API_KEY` under **Advanced settings → Secrets**
5. Click **Deploy** — you'll get a public URL in ~2 minutes

---

## 📁 Project Structure

```
ai-content-analyst/
├── app.py   # Main application (single file)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore
└── README.md
```

---

## 🔮 Roadmap

- [ ] PDF file upload support
- [ ] Streaming LLM responses (word-by-word output)
- [ ] Persistent storage across sessions
- [ ] Docker deployment support
- [ ] Support for additional languages

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 🙋 Author

Built by **Muneeba** ·  · [GitHub](https://github.com/muneebaumarr)

> ⭐ Star this repo if you found it useful!
