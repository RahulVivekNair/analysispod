from celery import Celery
import redis
import logging as logger
app = Celery('tasks', broker='redis://localhost:6380')

@app.task
def run_workflow(workflow_name, video_path, output_path):
    logger.info(f"Running workflow: {workflow_name}")
    logger.info(f"Video path: {video_path}")
    logger.info(f"Output path: {output_path}")
    if workflow_name == 'metadata':
        from extract_metadata import extract_metadata
        
        extract_metadata(video_path, output_path)
    # Add more workflow conditions as needed

    print(f"Workflow {workflow_name} completed for {video_path}")