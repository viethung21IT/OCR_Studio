import os
import io
import base64
import logging
from pathlib import Path
from PIL import Image

# ZeroGPU support for Hugging Face Spaces (optional)
try:
    import spaces
except ImportError:
    class MockSpaces:
        @staticmethod
        def GPU(func=None, duration=None):
            if func is None:
                def decorator(f):
                    return f
                return decorator
            return func
    spaces = MockSpaces()

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from ocr_engine import OCREngine

logger = logging.getLogger("App")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = BASE_DIR / "sample_images"

# ── Global OCR Engine ──────────────────────────────────────────────────
engine: OCREngine = None

def get_or_init_engine() -> OCREngine:
    global engine
    if engine is None:
        engine = OCREngine(vietocr_model_name="vgg_transformer", gpu_id=0)
    return engine


# ── Gradio predict function ───────────────────────────────────────────
@spaces.GPU
def gradio_predict(img, det_thresh, min_conf, upscale, adapt_pad, contrast, norm, beam):
    try:
        if img is None:
            return None, "Vui lòng chọn hoặc tải lên một hình ảnh.", {}
        eng = get_or_init_engine()
        res = eng.predict(
            image_input=img,
            det_thresh=float(det_thresh),
            min_confidence=float(min_conf),
            upscale_small=bool(upscale),
            adaptive_padding=bool(adapt_pad),
            contrast_boost=bool(contrast),
            normalize_text=bool(norm),
            use_beamsearch=bool(beam),
        )

        ann_img = None
        if "annotated_image_base64" in res and res["annotated_image_base64"]:
            b64 = res["annotated_image_base64"]
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[1]
            img_data = base64.b64decode(b64)
            pil_img = Image.open(io.BytesIO(img_data))
            ann_img = pil_img.copy()

        lines = [f"[{item.get('rec_confidence', 0.0)*100:.1f}%] {item.get('text', '')}" for item in res.get("boxes", [])]
        text_summary = "\n".join(lines) if lines else res.get("full_text", "Không phát hiện thấy chữ trong ảnh.")
        return ann_img, text_summary, res
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e


# ── Gradio UI Definition ──────────────────────────────────────────────
with gr.Blocks(title="AIC OCR Studio — Vietnamese OCR") as demo:
    gr.Markdown("# 🚀 AIC OCR Studio — Trích Xuất Chữ Tiếng Việt & Bounding Box")
    gr.Markdown("Nhận diện ký tự tiếng Việt siêu tốc bảo toàn 100% dấu thanh âm học phức tạp bằng mô hình DBNet & VietOCR Transformer.")

    with gr.Row():
        with gr.Column(scale=1):
            gr_input = gr.Image(type="filepath", label="Tải ảnh lên (Hoặc dán ảnh Ctrl+V)")
            with gr.Accordion("⚙️ Cấu hình Nhận diện nâng cao", open=False):
                det_th = gr.Slider(0.05, 0.90, value=0.25, step=0.05, label="Detection Threshold")
                min_cf = gr.Slider(0.05, 0.90, value=0.20, step=0.05, label="Min Confidence")
                up_sm = gr.Checkbox(value=True, label="Upscale chữ nhỏ")
                ad_pd = gr.Checkbox(value=True, label="Adaptive Polygon Expansion (Giữ trọn dấu tiếng Việt)")
                ct_bs = gr.Checkbox(value=False, label="CLAHE Contrast Boost")
                nm_tx = gr.Checkbox(value=True, label="NLP Vietnamese Normalizer")
                bm_sc = gr.Checkbox(value=False, label="Beam Search")
            gr_btn = gr.Button("🔍 Bắt đầu Nhận diện OCR", variant="primary", size="lg")

            # Example chips
            sample_candidates = [
                str(SAMPLES_DIR / "sample_billboard.jpg"),
                str(SAMPLES_DIR / "sample_news.jpg"),
                str(SAMPLES_DIR / "sample_subtitles.jpg"),
            ]
            existing_samples = [s for s in sample_candidates if Path(s).exists()]
            if existing_samples:
                gr.Examples(examples=existing_samples, inputs=gr_input, label="💡 Ảnh Mẫu Thử Nghiệm Nhanh")

        with gr.Column(scale=1):
            gr_ann = gr.Image(type="pil", label="Ảnh phát hiện Bounding Box Dạ quang")
            gr_text = gr.Textbox(label="Văn bản trích xuất được (kèm Độ tự tin)", lines=7)
            gr_json = gr.JSON(label="Chi tiết toạ độ Polygon Bounding Box")

    gr_btn.click(
        fn=gradio_predict,
        inputs=[gr_input, det_th, min_cf, up_sm, ad_pd, ct_bs, nm_tx, bm_sc],
        outputs=[gr_ann, gr_text, gr_json]
    )


# ── FastAPI Application ───────────────────────────────────────────────
app = FastAPI(
    title="AIC OCR Studio",
    description="Vietnamese OCR with DBNet (ONNX) and VietOCR Transformer",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for Dark Mode Web Studio
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.on_event("startup")
def startup_event():
    try:
        get_or_init_engine()
    except Exception as e:
        logger.warning(f"Engine deferred startup initialization: {e}")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the Dark Mode Web Studio at root."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found in static folder.")
    return FileResponse(str(index_path))

@app.get("/api/health")
async def health_check():
    try:
        eng = get_or_init_engine()
        return {
            "status": "ready",
            "device": eng.device_name,
            "model": eng.vietocr_model_name,
            "detector_loaded": eng.detector is not None,
            "providers": eng.detector.active_providers if eng.detector else [],
            "vietocr_loaded": eng.vietocr_predictor is not None
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
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
        logger.error(f"Error in /api/ocr: {e}", exc_info=True)
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
    sample_path = next((c for c in candidates if c.exists()), None)
    if not sample_path:
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
        logger.error(f"Error in /api/sample: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Mount Gradio at /gradio ───────────────────────────────────────────
app = gr.mount_gradio_app(app, demo, path="/gradio")


# ── Entrypoint ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    # If explicitly running on Hugging Face Spaces (PORT=7860), bind to 0.0.0.0
    if os.environ.get("SPACE_ID"):
        host = "0.0.0.0"
        port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host=host, port=port)
