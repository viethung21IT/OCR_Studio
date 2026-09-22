import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
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

def get_or_init_engine() -> OCREngine:
    global engine
    if engine is None:
        engine = OCREngine(vietocr_model_name="vgg_transformer", gpu_id=0)
    return engine

@app.on_event("startup")
def startup_event():
    get_or_init_engine()

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
    eng = get_or_init_engine()
    return {
        "status": "ready",
        "device": eng.device_name,
        "model": eng.vietocr_model_name,
        "detector_loaded": eng.detector is not None,
        "providers": eng.detector.active_providers if eng.detector else [],
        "vietocr_loaded": eng.vietocr_predictor is not None
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
    eng = get_or_init_engine()

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        results = eng.predict(
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
    eng = get_or_init_engine()

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
        results = eng.predict(
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


# ---------------------------------------------------------
# Gradio Companion Interface (for Hugging Face Spaces SDK)
# ---------------------------------------------------------
def gradio_predict(img):
    if img is None:
        return "Vui lòng chọn hoặc tải lên một hình ảnh.", {}
    eng = get_or_init_engine()
    res = eng.predict(img)
    lines = [f"[{item.get('rec_confidence', 0.0)*100:.1f}%] {item.get('text', '')}" for item in res.get("boxes", [])]
    text_summary = "\n".join(lines) if lines else res.get("full_text", "Không phát hiện thấy chữ trong ảnh.")
    return text_summary, res


with gr.Blocks(title="AIC OCR Studio - Vietnamese OCR") as demo:
    gr.Markdown("# 🚀 AIC OCR Studio — Trích Xuất Chữ Tiếng Việt")
    gr.Markdown(
        "💡 **Gợi ý**: Giao diện Studio đầy đủ với Bounding Box dạ quang tương tác và thanh công cụ đang chạy tại: **[👉 Nhấn vào đây để mở Web Studio](/)**"
    )
    with gr.Row():
        with gr.Column():
            gr_input = gr.Image(type="filepath", label="Tải ảnh lên (Hoặc dán ảnh)")
            gr_btn = gr.Button("🔍 Bắt đầu Nhận diện", variant="primary")
        with gr.Column():
            gr_text = gr.Textbox(label="Văn bản đã nhận diện (kèm Confidence)", lines=8)
            gr_json = gr.JSON(label="Dữ liệu chi tiết & Toạ độ Polygon Bounding Box")

    gr_btn.click(fn=gradio_predict, inputs=[gr_input], outputs=[gr_text, gr_json])

# Mount Gradio app into FastAPI
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("app:app", host=host, port=port, reload=False)

