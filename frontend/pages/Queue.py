import streamlit as st
import requests
st.header("Job Status")
job_status_response = requests.get("http://localhost:8000/job_status")
job_statuses = job_status_response.json()

for job_id, status in job_statuses.items():
    st.write(f"Job ID: {job_id}, Status: {status}")