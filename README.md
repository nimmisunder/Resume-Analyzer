<<<<<<< HEAD
# LangChain Resume Analyzer (RAG)

A local resume-analysis portfolio project. It compares a PDF resume with a pasted job description using a LangChain retrieval pipeline.

## Features

- PDF loading with `PyPDFLoader`
- Chunking with `RecursiveCharacterTextSplitter`
- Embeddings with Hugging Face `HuggingFaceEmbeddings`
- Vector indexing and similarity retrieval with FAISS
- Prompt construction with `ChatPromptTemplate`
- LCEL chain: `prompt | ChatOllama | StrOutputParser`
- Structured JSON feedback: matching skills, skill gaps, strengths, suggestions, summary rewrite and interview questions
- Streamlit user interface

## Architecture

```text
PDF Resume
   |
PyPDFLoader
   |
RecursiveCharacterTextSplitter
   |
HuggingFaceEmbeddings
   |
FAISS Vector Store
   |
Retriever <--- Job Description query
   |
Retrieved resume evidence + Job Description
   |
ChatPromptTemplate
   |
ChatOllama (local LLM)
   |
StrOutputParser -> JSON parsing
   |
Streamlit results dashboard
```

## Requirements

- Python 3.10+
- Ollama installed and running
- A model available locally (example: `llama3.1`)
- Internet connection for the first download of the embedding model and Python packages

## Setup on Windows PowerShell

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
ollama pull llama3.1
```

Start Ollama if it is not already running:

```powershell
ollama serve
```

In another terminal, run:

```powershell
streamlit run app.py
```

Upload a text-based PDF resume, paste a job description and click **Analyze with LangChain**.

## LangChain concepts used

| Concept | Class / API | Role |
|---|---|---|
| Document loader | `PyPDFLoader` | Extract PDF page text |
| Text splitter | `RecursiveCharacterTextSplitter` | Create overlapping chunks |
| Embedding integration | `HuggingFaceEmbeddings` | Convert chunks to vectors |
| Vector store | `FAISS` | Index and search vectors |
| Retriever | `as_retriever()` / `invoke()` | Fetch relevant resume chunks |
| Prompt | `ChatPromptTemplate` | Format the analysis instructions |
| Model | `ChatOllama` | Generate analysis locally |
| Output parser | `StrOutputParser` | Convert model output to text |
| LCEL | `prompt | llm | parser` | Compose the chain |

## Limitations and responsible use

- The match score is an approximate evidence-alignment estimate, not a real ATS score or hiring prediction.
- Scanned/image-only PDFs need OCR; this starter project handles text-based PDFs.
- The LLM can make mistakes. Verify all findings and never add skills or experience that are not true.
- Resume text is sent to the local Ollama instance configured in the app. Do not switch to a hosted model without considering privacy and consent.
- This starter version creates an in-memory FAISS index for each analysis; it does not persist resumes or analysis history.
- For production, add file-size limits, authentication, secure deletion/retention controls, tests, logging without sensitive resume content, and evaluation datasets.

## Suggested next steps

1. Add OCR for scanned resumes.
2. Add hybrid retrieval (BM25 + FAISS) and reranking.
3. Add section-aware parsing for skills, experience, education and projects.
4. Add evaluation examples and compare retrieval quality.
5. Add FastAPI endpoints and a database for user-managed history.
=======
# Resume-Analyzer
LangChain RAG implementation
>>>>>>> e57fe0129bfd37282aed191fc8d7cd068f5eeb15
