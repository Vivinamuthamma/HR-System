import re
from pdfminer.high_level import extract_text
import spacy
from spacy.matcher import Matcher
import streamlit as st
import pandas as pd
import os
from io import StringIO
from sentence_transformers import SentenceTransformer, util

# Initialize spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# CSV file to store candidate data
CSV_FILE = 'candidates.csv'

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def extract_contact_number_from_resume(text):
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    return match.group() if match else "N/A"

def extract_email_from_resume(text):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    return match.group() if match else "N/A"

def extract_skills_from_resume(text, skills_list):
    skills = []
    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)
    return skills

def extract_education_from_resume(text):
    education = []
    pattern = r"(?i)(?:Bsc|\bB\.\w+|\bM\.\w+|\bPh\.D\.\w+|\bBachelor(?:'s)?|\bMaster(?:'s)?|\bPh\.D)\s(?:\w+\s)*\w+"
    matches = re.findall(pattern, text)
    for match in matches:
        education.append(match.strip())
    return education

def extract_name(resume_text):
    matcher = Matcher(nlp.vocab)
    patterns = [
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}],
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}],
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}]
    ]
    for pattern in patterns:
        matcher.add('NAME', patterns=[pattern])
    doc = nlp(resume_text)
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        return span.text
    return "N/A"

def extract_job_title(text):
    # Heuristic: Look for likely job titles in the first 10 lines
    job_keywords = ['engineer', 'developer', 'manager', 'analyst', 'consultant', 'designer', 'lead', 'specialist', 'officer', 'director', 'intern']
    lines = text.split('\n')
    for line in lines[:10]:
        for keyword in job_keywords:
            if keyword.lower() in line.lower():
                return line.strip()
    return "N/A"

def extract_address(text):
    # Heuristic: Look for lines with numbers and street keywords
    address_keywords = ['street', 'st.', 'avenue', 'ave.', 'road', 'rd.', 'block', 'lane', 'ln.', 'drive', 'dr.', 'city', 'state', 'zip']
    lines = text.split('\n')
    for line in lines:
        if any(word in line.lower() for word in address_keywords) and re.search(r'\d', line):
            return line.strip()
    return "N/A"

def extract_linkedin_website(text):
    match = re.search(r'(https?://[^\s,;]+)', text)
    return match.group(1) if match else "N/A"

def extract_candidate_details(uploaded_file):
    # Extract text from uploaded file
    text = ""
    if uploaded_file.type == "application/pdf":
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        text = extract_text_from_pdf(tmp_file_path)
        os.remove(tmp_file_path)
    elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        text = extract_text_from_word(uploaded_file)
    else:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        text = stringio.read()

    if not text:
        return ("N/A",) * 7

    name = extract_name(text)
    job_title = extract_job_title(text)
    address = extract_address(text)
    phone = extract_contact_number_from_resume(text)
    linkedin_website = extract_linkedin_website(text)
    email = extract_email_from_resume(text)
    skills_list = ['Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management', 'Deep Learning', 'SQL', 'Tableau']
    skills = extract_skills_from_resume(text, skills_list)
    skills_str = ", ".join(skills) if skills else "N/A"

    return name, job_title, address, phone, linkedin_website, email, skills_str

def extract_text_from_word(uploaded_file):
    try:
        import docx2txt
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        text = docx2txt.process(tmp_file_path)
        os.remove(tmp_file_path)
        return text
    except ImportError:
        st.error("docx2txt is required to process Word files. Please install it.")
        return None

def save_candidates(candidates_df):
    candidates_df.to_csv(CSV_FILE, index=False)

def main():
    st.title("HR System")

    job_desc_file = st.file_uploader("Upload Job Description Document (txt, pdf, doc, docx)", type=['txt', 'pdf', 'doc', 'docx'])
    job_description = ""

    if job_desc_file:
        if job_desc_file.type == "application/pdf":
            job_description = extract_text_from_pdf(job_desc_file)
            if job_description is None:
                return
        elif job_desc_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            job_description = extract_text_from_word(job_desc_file)
            if job_description is None:
                return
        else:
            stringio = StringIO(job_desc_file.getvalue().decode("utf-8"))
            job_description = stringio.read()
    else:
        job_description = st.text_area("Or enter Job Description text", height=150)

    uploaded_files = st.file_uploader("Upload Resumes (txt, pdf, doc, docx)", accept_multiple_files=True, type=['txt', 'pdf', 'doc', 'docx'])

    if st.button("Process Resumes"):
        if not job_description:
            st.warning("Please provide a job description either by uploading a document or entering text.")
            return
        if not uploaded_files:
            st.warning("Please upload at least one resume.")
            return

        job_emb = model.encode(job_description, convert_to_tensor=True)
        new_candidates = []

        for uploaded_file in uploaded_files:
            name, job_title, address, phone, linkedin_website, email, skills = extract_candidate_details(uploaded_file)

            # Compute similarity score
            text = ""
            if uploaded_file.type == "application/pdf":
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                text = extract_text_from_pdf(tmp_file_path)
                os.remove(tmp_file_path)
            elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                text = extract_text_from_word(uploaded_file)
            else:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                text = stringio.read()

            if not text:
                continue

            resume_emb = model.encode(text, convert_to_tensor=True)
            score = util.pytorch_cos_sim(job_emb, resume_emb).item()
            score = round(score * 100, 2)

            new_candidates.append({
                'Resume File': uploaded_file.name,
                'Name': name,
                'Job Title': job_title,
                'Address': address,
                'Phone': phone,
                'LinkedIn/Website': linkedin_website,
                'Email': email,
                'Skills': skills,
                'Score': score
            })

        candidates_df = pd.DataFrame(new_candidates)
        candidates_df.reset_index(drop=True, inplace=True)
        candidates_df.index += 1  # Start serial numbers from 1
        candidates_df.insert(0, 'S.No', candidates_df.index)
        save_candidates(candidates_df)
        st.success("Resumes processed and data saved.")
    else:
        st.error("No candidate data to save.")

    st.subheader("Candidate Matches")
    if 'candidates_df' in locals() and not candidates_df.empty:
        st.dataframe(candidates_df.sort_values(by='Score', ascending=False))
    else:
        st.write("No candidate data available yet.")

if __name__ == "__main__":
    main()
