import os
import sys
import json
import traceback
import asyncio
import time
import uuid
from pathlib import Path
from typing import Optional, List
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, status
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# ========== CONFIGURATION ==========
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# OCR Import
try:
    from ocr.train import enhanced_ktp_processing, start_worker_process, get_worker_process, global_cleanup
except ImportError:
    sys.path.insert(0, str(BASE_DIR))
    from train import enhanced_ktp_processing, start_worker_process, get_worker_process, global_cleanup

# ========== FASTAPI APP ==========
app = FastAPI(
    title="KTP OCR Extraction API",
    description="API for extracting data from Indonesian ID cards (KTP)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ========== MIDDLEWARE ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# ========== SIMPLE RATE LIMITING ==========
class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < 60
        ]
        
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False
        
        self.requests[client_ip].append(now)
        return True

rate_limiter = RateLimiter(requests_per_minute=30)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path in ["/health", "/api/health"]:
        return await call_next(request)
    
    client_ip = request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Rate limit exceeded",
                "message": f"Maximum {rate_limiter.requests_per_minute} requests per minute",
                "retry_after": 60
            }
        )
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Log only API calls (not static files)
    if not request.url.path.startswith(("/static", "/uploads", "/favicon.ico")):
        print(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response

# ========== TEMPLATES & STATIC FILES ==========
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ========== CONCURRENCY CONTROL ==========
MAX_CONCURRENT_REQUESTS = 3
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# ========== APPLICATION STATE ==========
class AppState:
    def __init__(self):
        self.worker_started = False
        self.total_requests = 0
        self.successful_requests = 0

app_state = AppState()

# ========== STARTUP/SHUTDOWN ==========
@app.on_event("startup")
async def startup():
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, start_worker_process)
        app_state.worker_started = True
        print(" OCR worker started successfully")
    except Exception as e:
        print(f" OCR worker startup failed: {e}")
        traceback.print_exc()

@app.on_event("shutdown")
async def shutdown():
    try:
        await asyncio.to_thread(global_cleanup)
        print(" OCR worker cleaned up")
    except Exception as e:
        print(f" Cleanup failed: {e}")

# ========== HELPER FUNCTIONS ==========
def allowed_file(filename: str) -> bool:
    if not filename:
        return False
    
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
    ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''
    return ext in allowed_extensions

def generate_unique_filename(original_name: str) -> str:
    unique_id = uuid.uuid4().hex[:8]
    safe_name = "".join(c for c in original_name if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
    return f"{unique_id}_{safe_name}"

async def save_uploaded_file(file: UploadFile) -> Path:
    """Save uploaded file and return its path"""
    unique_filename = generate_unique_filename(file.filename)
    file_path = UPLOADS_DIR / unique_filename
    
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file"
        )
    
    if len(contents) > 5 * 1024 * 1024:  # 5MB
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Maximum size: 5MB"
        )
    
    file_path.write_bytes(contents)
    return file_path

async def process_ocr_file(file_path: Path, original_filename: str) -> dict:
    """
    Process file with OCR and return result.
    UPDATED: now supports new worker output format (raw_text as full lines).
    """

    # Run enhanced KTP processing
    text_lines, extracted_data, output_json = await asyncio.to_thread(
        enhanced_ktp_processing,
        str(file_path)
    )

    # Worker now returns output_json as str → convert to dict
    try:
        result = json.loads(output_json)
    except Exception:
        # fallback if something unexpected happens
        result = {
            "raw_text": text_lines,
            "extracted_data": extracted_data,
            "confidence_info": {"mean_confidence": 0}
        }

    # Add additional metadata
    result["metadata"] = {
        "filename": original_filename,
        "processed_id": file_path.name.split("_")[0],
        "file_size": file_path.stat().st_size,
        "timestamp": time.time(),
        "image_url": f"/uploads/{file_path.name}"
    }

    result["success"] = True
    return result


# ========== ROUTES ==========
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ocr", status_code=status.HTTP_200_OK)
async def process_ktp_legacy(file: UploadFile = File(...)):
    """Legacy endpoint - keep for backward compatibility"""
    return await process_ktp_internal(file)

@app.post("/api/ocr", status_code=status.HTTP_200_OK)
async def process_ktp(file: UploadFile = File(...)):
    """New endpoint with API prefix"""
    return await process_ktp_internal(file)

async def process_ktp_internal(file: UploadFile):
    """Internal processing function used by both endpoints"""
    # Validate file
    if not file.filename or not allowed_file(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Supported: PNG, JPG, JPEG, BMP, WEBP"
        )
    
    # Update request counter
    app_state.total_requests += 1
    
    # Save uploaded file
    try:
        file_path = await save_uploaded_file(file)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )
    
    # Process with concurrency control
    async with _semaphore:
        try:
            result = await process_ocr_file(file_path, file.filename)
            app_state.successful_requests += 1
            return JSONResponse(content=result)
            
        except Exception as e:
            # Clean up file on error
            if file_path.exists():
                file_path.unlink()
            
            error_response = {
                "success": False,
                "error": str(e),
                "timestamp": time.time(),
                "metadata": {
                    "filename": file.filename,
                    "processed_id": file_path.name.split('_')[0]
                }
            }
            
            if app.debug:
                error_response["traceback"] = traceback.format_exc()
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_response
            )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check_legacy():
    """Legacy health check endpoint"""
    return await health_check()

@app.get("/api/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    worker_status = "running"
    
    try:
        proc = await asyncio.to_thread(get_worker_process)
        if proc is None or proc.poll() is not None:
            worker_status = "stopped"
    except Exception:
        worker_status = "error"
    
    return {
        "status": "healthy" if worker_status == "running" else "degraded",
        "timestamp": time.time(),
        "worker": worker_status,
        "requests": {
            "total": app_state.total_requests,
            "successful": app_state.successful_requests,
            "success_rate": (
                app_state.successful_requests / app_state.total_requests
                if app_state.total_requests > 0 else 0
            )
        },
        "service": "ktp-ocr-api",
        "version": "1.0.0"
    }

@app.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    file_path = UPLOADS_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    return FileResponse(
        path=file_path,
        filename=filename
    )

@app.get("/favicon.ico")
async def favicon():
    """Return a placeholder favicon to avoid 404 errors"""
    return JSONResponse(content={"message": "No favicon configured"})

# ========== MAIN ENTRY POINT ==========
if __name__ == "__main__":
    import uvicorn
    
    app.startup_time = time.time()
    
    config = {
        "app": "app:app",
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", "8000")),
        "reload": os.getenv("ENV", "development") == "development",
        "log_level": "info",
        "workers": 1,
    }
    
    print(f"Starting KTP OCR API on {config['host']}:{config['port']}")
    print(f" Upload directory: {UPLOADS_DIR}")
    print(f" Environment: {os.getenv('ENV', 'development')}")
    print(f" Available endpoints:")
    print(f"   POST /ocr         - Process KTP image (legacy)")
    print(f"   POST /api/ocr     - Process KTP image")
    print(f"   GET  /health      - Health check (legacy)")
    print(f"   GET  /api/health  - Health check")
    print(f"   GET  /            - Upload form")
    
    uvicorn.run(**config)