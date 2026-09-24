# ── ZeroGPU: spaces MUST be the very first import ─────────────────────
# The spaces library monkey-patches torch.cuda so it must run before
# anything else touches CUDA. No try/except wrapper allowed here on HF.
try:
    import spaces  # real ZeroGPU on Hugging Face Spaces
    _HAS_SPACES = True
except ImportError:
    # Running locally: provide a no-op stub that preserves decorator API
    class _SpacesStub:
        @staticmethod
        def GPU(func=None, duration=None):
            if func is None:
                def decorator(f):
                    return f
                return decorator
            return func
    spaces = _SpacesStub()  # type: ignore
    _HAS_SPACES = False

import os
import io
import base64
import logging
from pathlib import Path

from PIL import Image
import gradio as gr

logger = logging.getLogger("App")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = BASE_DIR / "sample_images"

# ── Global OCR Engine (lazy) ───────────────────────────────────────────
# OCREngine is imported lazily so that torch/CUDA are only loaded after
# spaces has patched the CUDA runtime on Hugging Face Spaces.
_engine = None

def get_or_init_engine():
    global _engine
    if _engine is None:
        from ocr_engine import OCREngine  # lazy import – AFTER spaces patch
        _engine = OCREngine(vietocr_model_name="vgg_transformer", gpu_id=0)
    return _engine


# ── Gradio predict – decorated with @spaces.GPU ───────────────────────
# ZeroGPU scans for this decorator at startup. It MUST exist at module
# level and reference the real spaces.GPU (or our no-op stub locally).
@spaces.GPU
def gradio_predict(img, det_thresh, min_conf, upscale, adapt_pad, contrast, norm, beam):
    """Run OCR on the supplied image and return annotated result."""
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
        if res.get("annotated_image_base64"):
            b64 = res["annotated_image_base64"]
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[1]
            img_data = base64.b64decode(b64)
            pil_img = Image.open(io.BytesIO(img_data))
            ann_img = pil_img.copy()

        lines = [
            f"[{item.get('rec_confidence', 0.0) * 100:.1f}%] {item.get('text', '')}"
            for item in res.get("boxes", [])
        ]
        text_summary = (
            "\n".join(lines) if lines
            else res.get("full_text", "Không phát hiện thấy chữ trong ảnh.")
        )
        return ann_img, text_summary, res

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e


# ── Gradio UI ─────────────────────────────────────────────────────────
with gr.Blocks(title="OCR Studio — Vietnamese OCR") as demo:
    gr.Markdown("# OCR Studio — Trích Xuất Chữ Tiếng Việt & Bounding Box")
    gr.Markdown(
        "Nhận diện ký tự tiếng Việt bằng mô hình DBNet & VietOCR Transformer."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr_input = gr.Image(type="filepath", label="Tải ảnh lên (Hoặc dán ảnh Ctrl+V)")
            with gr.Accordion("⚙️ Cấu hình Nhận diện nâng cao", open=False):
                det_th = gr.Slider(0.05, 0.90, value=0.25, step=0.05, label="Detection Threshold")
                min_cf = gr.Slider(0.05, 0.90, value=0.20, step=0.05, label="Min Confidence")
                up_sm = gr.Checkbox(value=True,  label="Upscale chữ nhỏ")
                ad_pd = gr.Checkbox(value=True,  label="Adaptive Polygon Expansion (Giữ trọn dấu tiếng Việt)")
                ct_bs = gr.Checkbox(value=False, label="CLAHE Contrast Boost")
                nm_tx = gr.Checkbox(value=True,  label="NLP Vietnamese Normalizer")
                bm_sc = gr.Checkbox(value=False, label="Beam Search")
            gr_btn = gr.Button("🔍 Bắt đầu Nhận diện OCR", variant="primary", size="lg")

            sample_candidates = [
                str(SAMPLES_DIR / "sample_billboard.jpg"),
                str(SAMPLES_DIR / "sample_news.jpg"),
                str(SAMPLES_DIR / "sample_subtitles.jpg"),
            ]
            existing_samples = [s for s in sample_candidates if Path(s).exists()]
            if existing_samples:
                gr.Examples(examples=existing_samples, inputs=gr_input,
                            label="💡 Ảnh Mẫu Thử Nghiệm Nhanh")

        with gr.Column(scale=1):
            gr_ann  = gr.Image(type="pil", label="Ảnh phát hiện Bounding Box Dạ quang")
            gr_text = gr.Textbox(label="Văn bản trích xuất được (kèm Độ tự tin)", lines=7)
            gr_json = gr.JSON(label="Chi tiết toạ độ Polygon Bounding Box")

    gr_btn.click(
        fn=gradio_predict,
        inputs=[gr_input, det_th, min_cf, up_sm, ad_pd, ct_bs, nm_tx, bm_sc],
        outputs=[gr_ann, gr_text, gr_json],
    )


# ── Local-only entrypoint ──────────────────────────────────────────────
# On Hugging Face Spaces (sdk: gradio), HF auto-discovers and launches
# the `demo` Blocks object above — do NOT call demo.launch() at module
# level or it will conflict with the managed server on port 7860.
#
# Locally: run `python app.py` to start Gradio on any free port,
# or use run_demo.bat / run_demo.ps1 for the full FastAPI Web Studio.
if __name__ == "__main__":
    demo.launch(show_error=True)
