import streamlit as st
import fitz
import PyPDF2
from dotenv import load_dotenv
import os
from openai import OpenAI
import json

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

st.set_page_config(page_title="AI Resume Analyzer ATS", layout="wide")
st.title("📄 AI Resume Analyzer")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your PDF resume", type=["pdf"])

# --- Job Requirements Input ---
job_description = st.text_area("Paste Job Description / Requirements here", height=150)

# --- PDF Extraction ---
def extract_text_pymupdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.warning(f"PyMuPDF extraction failed: {e}")
        return ""

def extract_text_pypdf2(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.warning(f"PyPDF2 extraction failed: {e}")
        return ""

# --- AI Resume Analysis ---
def analyze_resume(text, job_desc):
    prompt = f"""
    You are an AI assistant. Analyze the following resume text:

    Resume:
    {text[:4000]}

    Job Description:
    {job_desc[:2000]}

    Tasks:
    1. Summarize the candidate's experience in 2-3 sentences.
    2. Extract key skills as a list.
    3. Compare candidate skills with job requirements and calculate a match percentage (0-100).
    4. List missing skills required for the job.
    5. Suggest improvements to make the resume stronger for this job.

    Respond ONLY in JSON format:
    {{
        "summary": "...",
        "skills": ["...", "..."],
        "match_percentage": 0,
        "missing_skills": ["...", "..."],
        "suggestions": ["...", "..."]
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=700
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return None

# --- Main Logic ---
if uploaded_file:
    st.subheader("Resume Preview 📑")

    # Extract text
    text = extract_text_pymupdf(uploaded_file)
    if not text.strip():
        uploaded_file.seek(0)
        text = extract_text_pypdf2(uploaded_file)

    st.text_area("Extracted Resume Text", text, height=300)

    if st.button("Analyze Resume for ATS Match"):
        if not job_description.strip():
            st.warning("Please enter the job description or requirements!")
        else:
            with st.spinner("Analyzing with AI..."):
                result = analyze_resume(text, job_description)

            if result:
                st.subheader("✅ AI Analysis Results")

                # Summary
                st.markdown("**Summary:**")
                st.info(result.get("summary", "No summary available"))

                # Skills
                st.markdown("**Skills:**")
                skills = result.get("skills", [])
                if skills:
                    st.write(", ".join(skills))
                else:
                    st.write("No skills detected.")

                # Match %
                match_percentage = result.get("match_percentage", 0)
                st.markdown(f"**ATS Match Percentage:** {match_percentage}%")
                st.progress(int(match_percentage))

                # Missing skills
                missing_skills = result.get("missing_skills", [])
                if missing_skills:
                    st.markdown("**Missing Skills / Keywords:**")
                    st.write(", ".join(missing_skills))
                else:
                    st.write("No critical skills missing!")

                # Suggestions
                st.markdown("**Suggestions:**")
                suggestions = result.get("suggestions", [])
                for s in suggestions:
                    st.write(f"- {s}")
