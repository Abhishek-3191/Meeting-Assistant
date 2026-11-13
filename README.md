# 🎙️ AI Meeting Assistant
<img width="1371" height="918" alt="image" src="https://github.com/user-attachments/assets/2a89157b-3793-430e-8547-002e16a92a41" />
## Live Website- 
https://meeting-assistant-y39zgmx6auwkmx4aufheko.streamlit.app/

An AI-powered meeting assistant that transforms audio and video recordings into structured meeting insights. It generates transcripts, summaries, action items, key decisions, and allows users to ask questions about their meetings using Retrieval-Augmented Generation (RAG).

Built with Python, Streamlit, OpenAI Whisper, Google Gemini, LangChain, and ChromaDB.

## ✨ Features

- **Audio & Video Processing:** Upload audio/video recordings or provide YouTube URLs.
- **Speech-to-Text:** Transcribe recordings using locally running OpenAI Whisper.
- **Multilingual Support:** Transcribe supported languages and optionally translate Hindi/Hinglish transcripts into English using Gemini.
- **AI Meeting Summaries:** Generate concise summaries of lengthy meetings using Gemini and a map-reduce approach.
- **Automatic Meeting Titles:** Generate meaningful titles from meeting content.
- **Action Item Extraction:** Identify tasks, assigned owners, and deadlines.
- **Key Decision Extraction:** Extract important decisions and unresolved questions.
- **AI Meeting Chat:** Ask questions about meeting content using RAG.
- **Semantic Search:** Retrieve relevant transcript sections using HuggingFace embeddings and ChromaDB.
- **Export Options:** Export meeting insights as PDF and TXT files.
- **Two Execution Modes:** Use the Streamlit interface or the command-line workflow.

## 🏗️ Architecture

The application follows a modular architecture with separate components for user interaction, audio processing, transcription, meeting analysis, and retrieval.

```text
AI-Meeting-Assistant/
│
├── app.py
├── main.py
├── test.py
├── test_gemini.py
│
├── utils/
│   └── audio_processor.py
│
├── core/
│   ├── rag_engine.py
│   ├── summarizer.py
│   ├── extractor.py
│   ├── transcriber.py
│   └── vector_store.py
│
├── downloads/
├── exports/
├── vector_db/
│
├── .env
├── .gitignore
├── Requirements.txt
├── pyproject.toml
└── uv.lock
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit user interface and application interaction |
| `main.py` | Command-line meeting processing workflow |
| `audio_processor.py` | Audio extraction, conversion, downloading, and chunking |
| `transcriber.py` | Local Whisper transcription and optional translation |
| `summarizer.py` | Meeting title generation and AI summarization |
| `extractor.py` | Action items, decisions, and questions extraction |
| `vector_store.py` | Embedding generation and ChromaDB vector storage |
| `rag_engine.py` | RAG chain construction, retrieval, and question answering |

## 🔄 Application Workflow

### 1. Audio Processing

Users upload an audio/video file or provide a YouTube URL.

The audio processor:
- Downloads audio when required.
- Extracts audio from video files.
- Converts audio into a suitable format.
- Splits long recordings into smaller chunks.

### 2. Speech-to-Text Transcription

The transcription module uses locally running OpenAI Whisper to convert audio chunks into text.

For Hindi/Hinglish recordings, Gemini can optionally translate the generated transcript into English.

The individual transcripts are combined into a complete meeting transcript.

### 3. Meeting Analysis

The complete transcript is processed by two independent modules.

**Summarization (`summarizer.py`)**
- Generates a meaningful meeting title.
- Produces a concise meeting summary.
- Uses a map-reduce approach to handle longer transcripts.

**Information Extraction (`extractor.py`)**
- Identifies action items.
- Extracts assigned owners and deadlines.
- Identifies key decisions.
- Extracts unresolved questions.

### 4. Vector Storage

The transcript is prepared for semantic retrieval.

1. Split the transcript into smaller text chunks.
2. Generate embeddings using the HuggingFace `all-MiniLM-L6-v2` model.
3. Store the embeddings and associated text in a local ChromaDB collection.

### 5. RAG-Based Meeting Chat

When a user asks a question:

1. Convert the question into an embedding.
2. Retrieve semantically relevant transcript chunks from ChromaDB.
3. Pass the retrieved context and question into the RAG prompt.
4. Use Gemini to generate a context-aware answer.
5. Display the answer in the Streamlit interface.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Frontend | Streamlit |
| Speech Recognition | OpenAI Whisper (local inference) |
| LLM | Google Gemini 2.5 Flash |
| LLM Framework | LangChain, LCEL |
| Embeddings | HuggingFace Sentence Transformers |
| Vector Database | ChromaDB |
| Audio Processing | FFmpeg, Pydub, yt-dlp |
| Export | ReportLab, TXT |

## ⚙️ Installation & Setup

### Prerequisites

- Python installed on your system.
- FFmpeg installed and accessible through PATH.
- Google Gemini API key.
- Git.

### 1. Clone the repository

```bash
git clone https://github.com/Abhishek-3191/Meeting-Assistant.git

cd Meeting-Assistant
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip

pip install -r Requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and add your Gemini API key.

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Use the environment variable name expected by your application code.

### 5. Install FFmpeg

Download FFmpeg from:

https://ffmpeg.org/download.html

Add the FFmpeg `bin` directory to your system PATH.

Verify the installation:

```bash
ffmpeg -version

ffprobe -version
```

### 6. Run the application

**Streamlit interface:**

```bash
streamlit run app.py
```

**Command-line workflow:**

```bash
python main.py
```

## 🧠 AI Models

### OpenAI Whisper

Whisper runs locally to perform speech recognition. This avoids requiring a paid cloud transcription API.

The transcription model is configured in `core/transcriber.py`.

### Google Gemini 2.5 Flash

Gemini is used for:
- Optional transcript translation.
- Meeting title generation.
- Meeting summarization.
- Action item and decision extraction.
- RAG-based question answering.

### HuggingFace Embeddings

The `sentence-transformers/all-MiniLM-L6-v2` model converts transcript chunks and user queries into vector representations for semantic retrieval.

The embedding model runs locally.

## 📁 Generated Data

The application creates local directories for processing and storage:

| Directory | Purpose |
|---|---|
| `downloads/` | Downloaded audio and processed audio chunks |
| `exports/` | Generated PDF and TXT exports |
| `vector_db/` | Persistent ChromaDB data |

These directories contain generated or potentially large files and should not be committed to Git.

## 🔐 Environment & Security

- Keep API keys in `.env`.
- Never commit `.env` or credentials to GitHub.
- Exclude virtual environments, downloaded recordings, generated exports, and local vector database files from version control.

## 🚀 Future Improvements

- Replace original Whisper inference with faster-whisper for improved transcription performance.
- Add asynchronous background processing for long recordings.
- Introduce FastAPI endpoints for a dedicated backend API.
- Add persistent meeting history and user-specific storage.
- Implement authentication and authorization.
- Add real-time transcription and meeting collaboration.
- Improve transcription caching and processing efficiency.

## 👨‍💻 Author

**Abhishek Srivastava**

- GitHub: [Abhishek-3191](https://github.com/Abhishek-3191)
- Project: [AI Meeting Assistant](https://github.com/Abhishek-3191/Meeting-Assistant)

---

*Built to make meeting conversations easier to understand, organize, search, and act upon.*
