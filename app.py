import os
from pathlib import Path

# ZeroGPU support for Hugging Face Spaces
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

from fastapi import File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from ocr_engine import OCREngine

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = BASE_DIR / "sample_images"

# Global OCR Engine instance
engine: OCREngine = None

def get_or_init_engine() -> OCREngine:
    global engine
    if engine is None:
        engine = OCREngine(vietocr_model_name="vgg_transformer", gpu_id=0)
    return engine


@spaces.GPU
def gradio_predict(img, det_thresh, min_conf, upscale, adapt_pad, contrast, norm, beam):
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
        import base64, io
        from PIL import Image
        b64 = res["annotated_image_base64"]
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        img_data = base64.b64decode(b64)
        ann_img = Image.open(io.BytesIO(img_data))
    
    lines = [f"[{item.get('rec_confidence', 0.0)*100:.1f}%] {item.get('text', '')}" for item in res.get("boxes", [])]
    text_summary = "\n".join(lines) if lines else res.get("full_text", "Không phát hiện thấy chữ trong ảnh.")
    return ann_img, text_summary, res


# Build Gradio UI with Dual Experience (Dark-mode studio iframe + Native Gradio Tester)
with gr.Blocks(title="AIC OCR Studio — Vietnamese OCR") as demo:
    with gr.Tabs():
        with gr.Tab("🌟 Web Studio (Dark Mode)"):
            gr.Markdown("### 🚀 AIC OCR Studio — Trích xuất chữ tiếng Việt & Bounding Box")
            gr.Markdown("💡 *Gợi ý: Bạn có thể mở giao diện toàn màn hình trực tiếp tại:* **[👉 /studio](/studio)**")
            gr.HTML('<iframe src="/studio" style="width: 100%; height: 950px; border: 1px solid rgba(255,255,255,0.1); border-radius: 12px;"></iframe>')
        
        with gr.Tab("🔍 Gradio Tester & API"):
            with gr.Row():
                with gr.Column():
                    gr_input = gr.Image(type="filepath", label="Tải ảnh lên (Hoặc dán ảnh Ctrl+V)")
                    with gr.Accordion("⚙️ Cấu hình Nhận diện nâng cao", open=False):
                        det_th = gr.Slider(0.05, 0.90, value=0.25, step=0.05, label="Detection Threshold")
                        min_cf = gr.Slider(0.05, 0.90, value=0.20, step=0.05, label="Min Confidence")
                        up_sm = gr.Checkbox(value=True, label="Upscale chữ nhỏ")
                        ad_pd = gr.Checkbox(value=True, label="Adaptive Polygon Expansion (Giữ trọn dấu tiếng Việt)")
                        ct_bs = gr.Checkbox(value=False, label="CLAHE Contrast Boost")
                        nm_tx = gr.Checkbox(value=True, label="NLP Vietnamese Normalizer")
                        bm_sc = gr.Checkbox(value=False, label="Beam Search")
                    gr_btn = gr.Button("🔍 Bắt đầu Nhận diện OCR", variant="primary")
                
                with gr.Column():
                    gr_ann = gr.Image(type="pil", label="Ảnh phát hiện Bounding Box")
                    gr_text = gr.Textbox(label="Văn bản trích xuất được", lines=6)
                    gr_json = gr.JSON(label="Chi tiết toạ độ Polygon Bounding Box")

            gr_btn.click(
                fn=gradio_predict,
                inputs=[gr_input, det_th, min_cf, up_sm, ad_pd, ct_bs, nm_tx, bm_sc],
                outputs=[gr_ann, gr_text, gr_json]
            )

# Mount FastAPI routes onto demo.app
app = demo.app

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/studio", response_class=HTMLResponse)
async def serve_studio():
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    demo.launch(server_name=host, server_port=port)
