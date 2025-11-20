from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import tempfile
import os
import json
import uuid
from datetime import datetime
from typing import List, Optional
import logging

from test import enhanced_ktp_processing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KTP OCR API",
    description="Advanced REST API for Indonesian ID Card (KTP) OCR Processing",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for batch processing (use database in production)
processing_queue = {}

class ProcessingResponse(BaseModel):
    request_id: str
    status: str
    filename: str
    processed_at: str
    data: Optional[dict] = None
    error: Optional[str] = None

class BatchProcessRequest(BaseModel):
    filenames: List[str]

@app.get("/")
async def root():
    return {
        "message": "KTP OCR API v2.0",
        "endpoints": {
            "health": "/health",
            "process": "/process-ktp (POST)",
            "batch_status": "/batch-status/{request_id} (GET)"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/process-ktp", response_model=ProcessingResponse)
async def process_ktp(file: UploadFile = File(...)):
    """
    Process single KTP image with enhanced response format
    """
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type. Allowed: {allowed_extensions}")
    
    temp_path = None
    request_id = str(uuid.uuid4())
    
    try:
        # Read and save file temporarily
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name
        
        logger.info(f"Processing KTP: {file.filename} (ID: {request_id})")
        
        # Process the image
        text_lines, normalized_data, output_json = enhanced_ktp_processing(temp_path, visualize=False)
        
        response = ProcessingResponse(
            request_id=request_id,
            status="success",
            filename=file.filename,
            processed_at=datetime.now().isoformat(),
            data={
                "raw_text": text_lines,
                "structured_data": normalized_data,
                "json_output": json.loads(output_json)
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        raise HTTPException(500, f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/process-ktp-batch")
async def process_ktp_batch(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Process multiple KTP images asynchronously
    """
    request_id = str(uuid.uuid4())
    processing_queue[request_id] = {
        "status": "processing",
        "total_files": len(files),
        "processed_files": 0,
        "results": [],
        "started_at": datetime.now().isoformat()
    }
    
    # Process files in background
    background_tasks.add_task(process_batch_files, request_id, files)
    
    return {
        "request_id": request_id,
        "status": "processing",
        "message": f"Processing {len(files)} files in background",
        "check_status": f"/batch-status/{request_id}"
    }

async def process_batch_files(request_id: str, files: List[UploadFile]):
    """Background task to process batch files"""
    for file in files:
        try:
            # Process each file (simplified - you might want to reuse process_ktp logic)
            temp_path = None
            try:
                contents = await file.read()
                file_extension = os.path.splitext(file.filename)[1].lower()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    temp_file.write(contents)
                    temp_path = temp_file.name
                
                text_lines, normalized_data, output_json = enhanced_ktp_processing(temp_path, visualize=False)
                
                processing_queue[request_id]["results"].append({
                    "filename": file.filename,
                    "status": "success",
                    "data": {
                        "raw_text": text_lines,
                        "structured_data": normalized_data,
                        "json_output": json.loads(output_json)
                    }
                })
                
            except Exception as e:
                processing_queue[request_id]["results"].append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })
            
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Error in batch processing for {file.filename}: {e}")
    
    # Update queue status
    processing_queue[request_id]["status"] = "completed"
    processing_queue[request_id]["completed_at"] = datetime.now().isoformat()

@app.get("/batch-status/{request_id}")
async def get_batch_status(request_id: str):
    """Check status of batch processing"""
    if request_id not in processing_queue:
        raise HTTPException(404, "Request ID not found")
    
    return processing_queue[request_id]

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )