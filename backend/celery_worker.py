from celery import Celery
import redis
import logging as logger
from extract_metadata import extract_metadata
from crowd_analysis import crowd_analysis
app = Celery('tasks', broker='redis://localhost:6380')

@app.task
def run_workflow(workflow_name, video_path, output_path,crowd_analysis_settings=None):
    logger.info(f"Running workflow: {workflow_name}")
    logger.info(f"Video path: {video_path}")
    logger.info(f"Output path: {output_path}")
    if workflow_name == 'metadata':

        extract_metadata(video_path, output_path)
    elif workflow_name == 'crowd_analysis':
        if crowd_analysis_settings:  # Use settings if provided
            # Implement your crowd analysis logic with these settings
            crowd_analysis( video_path,output_path,crowd_analysis_settings)
    # Add more workflow conditions as needed

    print(f"Workflow {workflow_name} completed for {video_path}")