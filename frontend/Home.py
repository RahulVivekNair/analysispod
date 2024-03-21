import streamlit as st
import requests
import datetime
import os
import json
import streamlit_authenticator as stauth
import yaml
with open('utils/pass.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
def display_home_after_login():
    st.title("AI Powered CCTV Footage Analysis")
    st.divider()

    # File upload
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    st.header("Settings")
    metadata_workflow = st.checkbox("Extract Metadata")
    crowd_analysis_workflow = st.checkbox("Crowd Analysis")

    crowd_analysis_settings = {}
    if crowd_analysis_workflow:
        with st.expander("Crowd Analysis Settings"):
            crowd_confdidence_threshold = st.slider("Crowd Confidence Threshold", 0.0, 1.0, 0.2, 0.05)
            crowd_bounding_box = st.checkbox("Bounding Boxes")
            crowd_heatmap = st.checkbox("Heatmap")
            crowd_trace = st.checkbox("Trace")
            crowd_label = st.checkbox("Label")
            crowd_analysis_settings = {
                "confidence_threshold": crowd_confdidence_threshold,
                "crowd_bounding_box": crowd_bounding_box,
                "crowd_heatmap": crowd_heatmap,
                "crowd_trace": crowd_trace,
                "crowd_label": crowd_label,
            }

    if st.button("Queue Analysis"):
        # Get the current timestamp
        uploaded_filename = uploaded_file.name
        filename_without_extension = os.path.splitext(uploaded_filename)[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a unique identifier for the job
        job_id = f"{timestamp}_{filename_without_extension}"
        # Create a list of selected workflows
        workflows = []
        if metadata_workflow:
            workflows.append("metadata")
        if crowd_analysis_workflow:
            workflows.append("crowd_analysis")

        # Send the file and workflow settings to the backend
        files = {"file": uploaded_file.getvalue()}
        data = {
            "job_id": job_id,
            "workflows": ",".join(workflows),
            "filename": uploaded_file.name,
            "crowd_analysis_settings": json.dumps(crowd_analysis_settings),
        }
        response = requests.post("http://localhost:8000/queue_job", files=files, data=data)
        if response.status_code == 200:
            st.success("Job queued successfully!")
        else:
            st.error("Failed to queue the job.")
name, authentication_status, username = authenticator.login('main')

if authentication_status:
    # Successful authentication
    display_home_after_login()  # Display regular home page content
    st.sidebar.button("Logout", on_click=authenticator.logout)  # Add a logout button
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
    # Optionally, you can hide other pages or elements here if not logged in
    # For example, you can conditionally display pages if `authentication_status` is True