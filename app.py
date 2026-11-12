import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from resume_pipeline import analyze_resume

st.set_page_config(page_title="LangChain Resume Analyzer", page_icon="📄", layout="wide")
st.title("📄 AI Resume Analyzer — LangChain")
st.caption("RAG-powered resume and job-description matching using LangChain, Hugging Face embeddings, FAISS and Ollama.")

with st.sidebar:
    st.header("LLM settings")
    ollama_base_url = st.text_input("Ollama URL", "http://localhost:11434")
    model_name = st.text_input("Ollama model", "llama3.1")
    top_k = st.slider("Resume chunks to retrieve", 2, 10, 5)
    st.caption("Run Ollama locally and pull the selected model before analysis.")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload resume PDF", type=["pdf"])
with col2:
    job_description = st.text_area("Paste job description", height=250)

if st.button("Analyze with LangChain", type="primary", use_container_width=True):
    if resume_file is None:
        st.error("Upload a PDF resume first.")
    elif not job_description.strip():
        st.error("Paste a job description first.")
    else:
        try:
            with st.spinner("Loading, chunking, embedding and retrieving resume content..."):
                result = analyze_resume(
                    resume_file=resume_file,
                    job_description=job_description,
                    ollama_base_url=ollama_base_url,
                    model_name=model_name,
                    top_k=top_k,
                )

            st.success("Analysis completed")
            st.metric("Estimated alignment", f"{result['match_score']}/100")
            st.caption("Indicative alignment only — not a genuine ATS score or hiring prediction.")

            left, right = st.columns(2)
            with left:
                st.subheader("Matching skills")
                for skill in result.get("matching_skills", []):
                    st.write(f"✓ {skill}")
                st.subheader("Strengths")
                for item in result.get("strengths", []):
                    st.write(f"• {item}")
            with right:
                st.subheader("Missing or weak skills")
                for skill in result.get("missing_skills", []):
                    st.write(f"• {skill}")
                st.subheader("Improvement areas")
                for item in result.get("improvement_areas", []):
                    st.write(f"• {item}")

            st.subheader("Resume improvement suggestions")
            for item in result.get("suggestions", []):
                st.write(f"• {item}")

            st.subheader("Suggested professional summary")
            st.info(result.get("summary_rewrite", ""))

            st.subheader("Interview questions")
            for item in result.get("interview_questions", []):
                st.write(f"• {item}")

            with st.expander("Retrieved resume context (RAG)"):
                for i, doc in enumerate(result["retrieved_documents"], start=1):
                    st.markdown(f"**Chunk {i} — page {doc['page']}**")
                    st.write(doc["content"])
            with st.expander("Raw LLM response"):
                st.code(result["raw_response"])
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.info("Check that Ollama is running, the model is pulled, and all dependencies are installed.")
