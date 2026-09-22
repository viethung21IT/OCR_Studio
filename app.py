import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ocr_engine import OCREngine

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = BASE_DIR / "sample_images"

app = FastAPI(title="AIC OCR Studio Web Demo", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global OCR Engine instance
engine: OCREngine = None

@app.on_event("startup")
def startup_event():
    global engine
    engine = OCREngine(vietocr_model_name="vgg_transformer", gpu_id=0)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path))

@app.get("/api/health")
async def health_check():
    if engine is None:
        return {"status": "starting", "device": "loading"}
    return {
        "status": "ready",
        "device": engine.device_name,
        "model": engine.vietocr_model_name,
        "detector_loaded": engine.detector is not None,
        "providers": engine.detector.active_providers if engine.detector else [],
        "vietocr_loaded": engine.vietocr_predictor is not None
    }

@app.post("/api/ocr")
async def run_ocr(
    file: UploadFile = File(...),
    det_thresh: float = Form(0.25),
    min_confidence: float = Form(0.20),
    model_name: str = Form("vgg_transformer"),
    upscale_small: bool = Form(True),
    adaptive_padding: bool = Form(True),
    contrast_boost: bool = Form(False),
    normalize_text: bool = Form(True),
    use_beamsearch: bool = Form(False),
):
    if engine is None:
        raise HTTPException(status_code=503, detail="OCR engine is still initializing.")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        results = engine.predict(
            image_input=image_bytes,
            det_thresh=det_thresh,
            min_confidence=min_confidence,
            vietocr_model=model_name,
            upscale_small=upscale_small,
            adaptive_padding=adaptive_padding,
            contrast_boost=contrast_boost,
            normalize_text=normalize_text,
            use_beamsearch=use_beamsearch,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sample/{sample_id}")
async def run_sample_ocr(
    sample_id: str,
    det_thresh: float = 0.25,
    min_confidence: float = 0.20,
    model_name: str = "vgg_transformer",
    upscale_small: bool = True,
    adaptive_padding: bool = True,
    contrast_boost: bool = False,
    normalize_text: bool = True,
    use_beamsearch: bool = False,
):
    if engine is None:
        raise HTTPException(status_code=503, detail="OCR engine is still initializing.")

    candidates = [
        SAMPLES_DIR / f"sample_{sample_id}.jpg",
        SAMPLES_DIR / f"hard_{sample_id}.jpg",
        SAMPLES_DIR / f"{sample_id}.jpg",
    ]
    sample_path = None
    for c in candidates:
        if c.exists():
            sample_path = c
            break

    if not sample_path or not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")

    try:
        results = engine.predict(
            image_input=str(sample_path),
            det_thresh=det_thresh,
            min_confidence=min_confidence,
            vietocr_model=model_name,
            upscale_small=upscale_small,
            adaptive_padding=adaptive_padding,
            contrast_boost=contrast_boost,
            normalize_text=normalize_text,
            use_beamsearch=use_beamsearch,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
