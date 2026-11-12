import json
import re
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def load_and_split_resume(pdf_path):
    """Load PDF pages and split them into overlapping LangChain Documents."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\\n\\n", "\\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)


def build_vector_store(chunks):
    """Create embeddings and index resume chunks in FAISS."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(chunks, embeddings)


def make_prompt():
    template = """
You are an expert resume reviewer. Compare the candidate's resume evidence with the job description.

Rules:
- Use only facts supported by the resume context.
- Never invent skills, years of experience, employers, education, certifications or achievements.
- Distinguish explicitly demonstrated skills from missing or weakly evidenced skills.
- Give practical, honest suggestions. Do not claim to predict hiring or ATS outcomes.
- Return ONLY valid JSON. Do not use Markdown fences.

Required JSON structure:
{{
  "match_score": 0,
  "matching_skills": ["skills supported by resume and relevant to JD"],
  "missing_skills": ["important JD skills not evidenced in resume"],
  "strengths": ["specific strengths supported by resume"],
  "improvement_areas": ["specific gaps or unclear areas"],
  "suggestions": ["actionable, truthful resume improvements"],
  "summary_rewrite": "A 2-4 sentence professional summary using only resume facts",
  "interview_questions": ["role-specific questions based on the JD and resume"]
}}

The match_score must be an integer from 0 to 100 and represents approximate evidence alignment only.

JOB DESCRIPTION:
{job_description}

RETRIEVED RESUME CONTEXT:
{resume_context}
"""
    return ChatPromptTemplate.from_template(template)


def parse_json_response(raw):
    """Parse JSON returned by the local model, tolerating occasional code fences."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned)
    cleaned = re.sub(r"\\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The LLM did not return a valid JSON object. Try again or use a different model.")
    data = json.loads(cleaned[start:end + 1])
    data["match_score"] = max(0, min(100, int(data.get("match_score", 0))))
    return data


def analyze_resume(resume_file, job_description, ollama_base_url, model_name, top_k=5):
    """End-to-end LangChain RAG workflow for resume/job-description analysis."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(resume_file.getvalue())
            temp_path = tmp.name

        chunks = load_and_split_resume(temp_path)
        if not chunks:
            raise ValueError("No readable text found in the PDF. Scanned PDFs require OCR.")

        vector_store = build_vector_store(chunks)
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        # Use the job description as the retrieval query to find the most relevant resume evidence.
        retrieved_docs = retriever.invoke(
            "Find resume experience, skills, projects, education and achievements relevant to this job: "
            + job_description[:5000]
        )
        resume_context = "\\n\\n".join(
            f"[Page {doc.metadata.get('page', 'unknown') + 1 if isinstance(doc.metadata.get('page'), int) else 'unknown'}]\\n{doc.page_content}"
            for doc in retrieved_docs
        )

        llm = ChatOllama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0,
        )
        prompt = make_prompt()
        chain = prompt | llm | StrOutputParser()

        raw_response = chain.invoke({
            "job_description": job_description[:9000],
            "resume_context": resume_context[:14000],
        })
        result = parse_json_response(raw_response)
        result["raw_response"] = raw_response
        result["retrieved_documents"] = [
            {
                "page": (doc.metadata.get("page", 0) + 1)
                    if isinstance(doc.metadata.get("page", 0), int) else "unknown",
                "content": doc.page_content,
            }
            for doc in retrieved_docs
        ]
        return result
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
