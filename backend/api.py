from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from extract_metadata import extract_metadata
from celery import Celery
from celery_worker import run_workflow
import os
app = FastAPI()

@app.post("/queue_job")
async def queue_job(
    file: UploadFile = File(...),
    job_id: str = Form(...),
    workflows: str = Form(...),
    filename: str = Form(...)
):
    # Create a directory for the job
    job_dir = os.path.join("uploads", job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Save the uploaded file
    file_path = os.path.join(job_dir, filename)
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    # Process the workflows
    selected_workflows = workflows.split(",")
    print(f"Selected workflows: {selected_workflows}")
    for workflow in workflows.split(","):
        output_path = os.path.join(job_dir, f"{workflow}_output.json")
        run_workflow(workflow,file_path, output_path)
    # Add your workflow execution logic here
    # ...

    return JSONResponse(content={"message": "Job queued successfully"})