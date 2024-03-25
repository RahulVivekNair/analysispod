import redis
import streamlit as st
import time
import json
r = redis.Redis(host='localhost', port=6380)  

def get_job_info():
    worker_names = r.client_list(type='normal')  # Get worker names
    jobs_info = []
    for worker_name in worker_names:
        worker_key = f'celery-task-meta-{worker_name["id"]}'
        jobs = r.lrange(worker_key, 0, -1)  # Get jobs for this worker
        for job in jobs:
            job_dict = json.loads(job.decode('utf-8'))
            jobs_info.append(job_dict)
    return jobs_info

def format_jobs_for_display(jobs_info):
    display_jobs = []
    for job in jobs_info:
        display_jobs.append({
            'job_id': job['args'][0],  # Assuming job ID is the first argument
            'status': job['status'],
            'workflow': job['args'][1],
            # Add more fields as needed
        })
    return display_jobs
def display_job_queues():
    st.title("Job Queues")

    with st.expander("Running Jobs"):
        job_display_placeholder = st.empty()  # Placeholder for updates

        while True:  # Periodic updates
            jobs_info = get_job_info()
            formatted_jobs = format_jobs_for_display(jobs_info)

            # Display formatted_jobs using a suitable Streamlit component (e.g., st.table) 
            job_display_placeholder.table(formatted_jobs) 

            time.sleep(5)  # Adjust refresh interval

display_job_queues()