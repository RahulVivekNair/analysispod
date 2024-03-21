import streamlit as st
import redis
import time

redis_client = redis.Redis(host='localhost', port=6380)

def display_job_status():
    st.title("Job Status")
    while True:  # Loop to periodically refresh
        jobs = redis_client.keys()  # Get all job IDs
        if jobs:
            for job_id in jobs:
                status = redis_client.get(job_id).decode()  
                st.write(f"Job ID: {job_id} - Status: {status}")
        else:
            st.write("No jobs found.")
        time.sleep(5)  # Adjust refresh interval

display_job_status() 