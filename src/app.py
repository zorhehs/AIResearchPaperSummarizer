import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/summarize"

st.set_page_config(page_title="AI Research Paper Summarizer", layout="wide")
st.title("AI Research Paper Summarizer")
st.write("Upload a PDF or paste a DOI to get a structured summary.")

tab1, tab2 = st.tabs(["Upload PDF", "Enter DOI"])

result = None

with tab1:
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_file and st.button("Summarize PDF"):
        with st.spinner("Processing... this can take 30-60 seconds for long papers."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                result = response.json()
            else:
                st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")

with tab2:
    doi_input = st.text_input("Enter a DOI (e.g. 10.1371/journal.pone.0121283)")
    if doi_input and st.button("Summarize DOI"):
        with st.spinner("Processing... this can take 30-60 seconds."):
            response = requests.post(API_URL, data={"doi": doi_input})
            if response.status_code == 200:
                result = response.json()
            else:
                st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")

if result:
    st.success(f"**{result['title']}**")
    st.caption(f"Source: {result['source']}")

    st.subheader("Summary")
    st.write(result["summary"])

    st.subheader("Methodology")
    st.write(result["methodology"])

    st.subheader("Research Gaps")
    st.write(result["research_gaps"])

    st.subheader("Findings")
    st.write(result["findings"])

    st.subheader("Future Work")
    st.write(result["future_work"])
