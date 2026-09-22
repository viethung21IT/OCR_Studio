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

import gradio as gr
from ocr_engine import OCREngine

BASE_DIR = Path(__file__).resolve().parent
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
            import base64, io
            from PIL import Image
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


# Build Clean & Elegant Gradio Studio
with gr.Blocks(title="AIC OCR Studio — Vietnamese OCR") as demo:
    gr.Markdown("# 🚀 AIC OCR Studio — Trích Xuất Chữ Tiếng Việt & Bounding Box (ZeroGPU)")
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    demo.launch(server_name=host, server_port=port, show_error=True)
