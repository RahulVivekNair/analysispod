import streamlit as st
import os
import json
import shutil

def delete_report(job_dir):
    if st.button(f"Delete {job_dir}"):
        if st.warning(f"Are you sure you want to delete the report for {job_dir}?"):
            shutil.rmtree(f"../backend/uploads/{job_dir}")
            st.toast(f"Report for {job_dir} deleted successfully.")
            st.rerun()

def display_reports_page():
    st.title("Reports")
    st.divider()

    # Get the list of directories in the "uploads" directory
    uploads_dir = "../backend/uploads"
    job_dirs = [dir for dir in os.listdir(uploads_dir) if os.path.isdir(os.path.join(uploads_dir, dir))]

    # Display the list of job directories
    if job_dirs:
        for job_dir in job_dirs:
            with st.expander(job_dir):
                st.header("Original Video")
                st.divider()
                st.video(f"../backend/uploads/{job_dir}/original.mp4")

                metadata_file = f"../backend/uploads/{job_dir}/metadata_output.json"
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r") as file:
                        metadata = file.read()
                    st.header("Video Metadata")
                    st.divider()
                    st.json(metadata)

                crowd_file = f"../backend/uploads/{job_dir}/crowd.json"
                if os.path.exists(crowd_file):
                    st.header("Crowd Analysis")
                    st.divider()
                    st.video(f"../backend/uploads/{job_dir}/crowd_analysis_output.mp4")
                    st.image(f"../backend/uploads/{job_dir}/crowd_count_graph.png")
                    with open(crowd_file, "r") as file:
                        crowd_data = json.load(file)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(label="Average Crowd Count", value=crowd_data["average_crowd_count"])
                    
                    with col2:
                        peak_times = ", ".join(str(time) for time in crowd_data["peak_times"])
                        st.metric(label="Peak Times (seconds)", value=peak_times)

                anomaly_file = f"../backend/uploads/{job_dir}/anomaly.json"
                if os.path.exists(anomaly_file):
                    with open(anomaly_file, "r") as file:
                        anomaly_data = json.load(file)
                    st.header("Anomaly Analysis")
                    st.divider()
                    st.video(f"../backend/uploads/{job_dir}/anomaly_analysis.mp4")
                    st.image(f"../backend/uploads/{job_dir}/anomaly_analysis_graph.png")
                    st.json(anomaly_data)
                    st.subheader("Anomaly Frames")
                    st.divider()
                    anomaly_frames_dir = f"../backend/uploads/{job_dir}/anomaly_frames"
                    anomaly_frames = os.listdir(anomaly_frames_dir)
                    for frame in anomaly_frames:
                        st.image(f"{anomaly_frames_dir}/{frame}")

                delete_report(job_dir)
    else:
        st.info("No job directories found.")

display_reports_page()