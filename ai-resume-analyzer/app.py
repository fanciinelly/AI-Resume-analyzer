# app.py

import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
import plotly.graph_objects as go
import json
import re
from io import BytesIO
from fpdf import FPDF
from datetime import datetime
import csv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

TOTAL_DAILY_LIMIT = 250000 # Updated to reflect the new model's limit
session_tokens_used = 0

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"Direct text extraction failed: {e}")

    try:
        images = convert_from_path(pdf_path)
        for image in images:
            page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
    except Exception as e:
        print(f"OCR failed: {e}")

    return text.strip()

def analyze_resume(resume_text, job_description=None):
    global session_tokens_used
    # **FIX 1: Updated model from 'gemini-1.5-flash' to 'gemini-2.5-flash'**
    model = genai.GenerativeModel("gemini-2.5-flash")
    base_prompt = f"""
    You are an experienced HR with technical expertise. Review the provided resume.
    Share a professional evaluation on whether the profile aligns with the role.
    Mention skills the candidate has, suggest improvements, and recommend relevant courses.

    Resume:
    {resume_text}
    """

    if job_description:
        base_prompt += f"""
        Additionally, compare this resume to the job description:
        {job_description}
        Highlight strengths and weaknesses in relation to the job.
        """

    session_tokens_used += len(base_prompt.split()) + 1000
    response = model.generate_content(base_prompt)
    return response.text.strip()

def calculate_scores(resume_text, job_description):
    global session_tokens_used
    # **FIX 2: Updated model from 'gemini-1.5-flash' to 'gemini-2.5-flash'**
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    Return a valid JSON with scores (1-10) for:
      - Technical Skills Match
      - Experience Relevance
      - Education Alignment
      - Career Progression
      - Overall Match

    Each must include:
      - "score": int
      - "reason": str
      - "suggestion": str (optional, if score < 8)

    Resume: {resume_text}
    Job Description: {job_description}

    Respond with only JSON. No markdown.
    """
    session_tokens_used += len(prompt.split()) + 500
    response = model.generate_content(prompt)
    raw_output = response.text.strip()
    try:
        cleaned = re.sub(r"^```json|```$", "", raw_output.strip())
        return json.loads(cleaned)
    except Exception as e:
        st.error("Failed to parse scores. Check Gemini output.")
        st.text_area("Raw Gemini Score Output", raw_output, height=250)
        return {}

def calculate_match_percentage(resume_text, job_description):
    global session_tokens_used
    # **FIX 3: Updated model from 'gemini-1.5-flash' to 'gemini-2.5-flash'**
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f'''
    Return JSON in this format only:
    {{"match": 85, "explanation": "Candidate meets most requirements."}}

    Resume: {resume_text}
    Job Description: {job_description}

    Respond with only JSON. No markdown.
    '''
    session_tokens_used += len(prompt.split()) + 300
    response = model.generate_content(prompt)
    raw_output = response.text.strip()
    try:
        cleaned = re.sub(r"^```json|```$", "", raw_output.strip())
        return json.loads(cleaned)
    except Exception as e:
        st.error("Failed to parse match percentage.")
        st.text_area("Raw Gemini Match Output", raw_output, height=250)
        return {"match": 0, "explanation": "Parsing error."}

def display_scores(scores):
    st.subheader("Resume Match Analysis")
    categories = list(scores.keys())
    values = [score["score"] for score in scores.values()]

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False
    )
    st.plotly_chart(fig)

    for category, details in scores.items():
        st.write(f"**{category}**: {details['score']}/10")
        st.write(f"- Reason: {details.get('reason', 'N/A')}")
        if details['score'] < 8:
            st.write(f"- Suggestion: {details.get('suggestion', 'N/A')}")

def export_report(analysis_text, scores, match):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Resume Analysis Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Overall Match: {match['match']}%", ln=True)
    pdf.multi_cell(0, 10, txt=match['explanation'].encode("latin-1", "replace").decode("latin-1"))
    pdf.ln(5)

    for cat, val in scores.items():
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=f"{cat}: {val['score']}/10", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=val['reason'].encode("latin-1", "replace").decode("latin-1"))
        if val['score'] < 8:
            pdf.multi_cell(0, 10, txt=val['suggestion'].encode("latin-1", "replace").decode("latin-1"))
        pdf.ln(2)

    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=analysis_text.encode("latin-1", "replace").decode("latin-1"))

    pdf.output("resume_analysis_report.pdf")
    with open("resume_analysis_report.pdf", "rb") as f:
        st.download_button(
            label="📅 Download Report",
            data=f,
            file_name="resume_analysis_report.pdf",
            mime="application/pdf"
        )

def save_history(filename, job_desc, match, scores):
    with open("history.csv", "a", newline="") as file:
        writer = csv.writer(file)
        row = [datetime.now(), filename, match['match']]
        for score in scores.values():
            row.append(score['score'])
        writer.writerow(row)

st.set_page_config(page_title="Resume Analyzer", layout="wide")

# Theme toggle
if "theme" not in st.session_state:
    st.session_state.theme = "light"

with st.sidebar:
    if st.button("Toggle Theme"):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
    st.markdown(f"### Current Theme: `{st.session_state.theme}`")
    st.markdown("---")
    st.markdown("📬 [Email](mailto:nwaforprincewill21@gmail.com)")
    st.markdown("💻 [GitHub](https://github.com/fanciinelly)")

st.title("AI Resume Analyzer")

with st.expander("ℹ️ About this Project"):
    st.markdown("""
    **Developer:** Nwafor Princewill  
    **Final Year Project - Computer Science** This project is designed to help **HR professionals**, **job seekers**, and **career coaches** by providing automated resume evaluations using **Google Gemini AI**. It:

    - Evaluates uploaded resumes and matches them with provided job descriptions.
    - Generates an in-depth analysis report with recommendations.
    - Calculates alignment scores and visualizes them.
    - Offers a downloadable PDF report.
    - Now includes live token usage tracking.
    """)

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
with col2:
    job_description = st.text_area("Enter Job Description:", placeholder="Paste the job description here...")

if uploaded_file:
    st.success("Resume uploaded successfully!")
    with open("uploaded_resume.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    resume_text = extract_text_from_pdf("uploaded_resume.pdf")

    if st.button("Analyze Resume"):
        with st.spinner("Analyzing resume..."):
            try:
                match_result = calculate_match_percentage(resume_text, job_description)
                scores = calculate_scores(resume_text, job_description)
                display_scores(scores)

                st.metric("Overall Match", f"{match_result['match']}%")
                st.caption(match_result['explanation'])

                analysis = analyze_resume(resume_text, job_description)
                st.success("Analysis complete!")
                st.write(analysis)

                export_report(analysis, scores, match_result)
                save_history(uploaded_file.name, job_description, match_result, scores)

                st.info(f"🔢 Tokens used this session: {session_tokens_used:,} / {TOTAL_DAILY_LIMIT:,}")
            except Exception as e:
                st.error(f"Analysis failed: {e}")
else:
    st.warning("Please upload a resume in PDF format.")

st.markdown("---")
st.markdown("""
<p style='text-align: center;'>Powered by <b>Streamlit</b> and <b>Google Gemini AI</b> | Developed by <b>Nwafor Princewill</b></p>
""", unsafe_allow_html=True)
