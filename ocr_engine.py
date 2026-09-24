import os
import io
import re
import time
import base64
import logging
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Dynamic configuration of Windows DLL paths for CUDA 12 / ONNX Runtime
def _setup_cuda_dlls():
    if os.name != "nt":
        return
    import sys
    import site

    candidates = []
    try:
        for sp in site.getsitepackages():
            candidates.append(Path(sp) / "torch" / "lib")
    except Exception:
        pass
    candidates.append(Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib")
    candidates.append(Path(sys.prefix) / "lib" / "site-packages" / "torch" / "lib")
    candidates.append(Path(r"G:\miniconda3\envs\CV_env\Lib\site-packages\torch\lib"))

    for torch_lib in candidates:
        if torch_lib.exists():
            try:
                os.add_dll_directory(str(torch_lib))
            except Exception:
                pass
            os.environ["PATH"] = str(torch_lib) + ";" + os.environ.get("PATH", "")
            break

_setup_cuda_dlls()

import torch
import onnxruntime as ort
import pyclipper
from shapely.geometry import Polygon

logger = logging.getLogger("OCREngine")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" / "det" / "ch_PP-OCRv4_det"
DEFAULT_ONNX_PATH = MODELS_DIR / "model.onnx"


def _patch_pkg_resources() -> None:
    """
    Inject a pkg_resources compatibility shim into sys.modules if the real
    module is unavailable.

    On Hugging Face ZeroGPU, code inside @spaces.GPU decorated functions runs
    with a restricted sys.path that may exclude the system site-packages even
    though setuptools is physically installed there.  VietOCR's Cfg class uses
    pkg_resources.resource_string() to load its YAML config files, so without
    this shim VietOCR cannot initialise.

    The shim implements only the two functions VietOCR actually uses:
      - resource_string(package, resource)  -> bytes
      - resource_filename(package, resource) -> str
    Both resolve the resource path via importlib.util, which is always available.
    """
    try:
        import pkg_resources  # noqa: F401  – already available, nothing to do
        return
    except ImportError:
        pass

    import sys
    import types
    import importlib.util

    def _find_module_dir(module_name: str) -> str:
        """Return the directory that contains the given dotted module."""
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin:
            return os.path.dirname(os.path.abspath(spec.origin))
        # Walk up the dotted hierarchy
        parts = module_name.split('.')
        for depth in range(len(parts) - 1, 0, -1):
            parent = '.'.join(parts[:depth])
            pspec = importlib.util.find_spec(parent)
            if pspec and pspec.submodule_search_locations:
                return list(pspec.submodule_search_locations)[0]
        return os.getcwd()

    def resource_string(package_or_requirement, resource_name: str) -> bytes:
        pkg = (
            package_or_requirement
            if isinstance(package_or_requirement, str)
            else str(package_or_requirement)
        )
        base_dir = _find_module_dir(pkg)
        # Primary candidate: resource relative to the module's own directory
        candidates = [os.path.join(base_dir, resource_name)]
        # Secondary: resource relative to the top-level package root
        #   e.g. 'vietocr.tool.config' → look in <vietocr_root>/config/...
        root_pkg = pkg.split('.')[0]
        if root_pkg != pkg:
            root_spec = importlib.util.find_spec(root_pkg)
            if root_spec and root_spec.origin:
                root_dir = os.path.dirname(os.path.abspath(root_spec.origin))
                candidates.append(os.path.join(root_dir, resource_name))
        for path in candidates:
            if os.path.exists(path):
                with open(path, 'rb') as fh:
                    return fh.read()
        raise FileNotFoundError(
            f"pkg_resources shim: resource '{resource_name}' not found for '{pkg}'.\n"
            f"Searched: {candidates}"
        )

    def resource_filename(package_or_requirement, resource_name: str) -> str:
        pkg = (
            package_or_requirement
            if isinstance(package_or_requirement, str)
            else str(package_or_requirement)
        )
        return os.path.join(_find_module_dir(pkg), resource_name)

    class MockDistribution:
        def __init__(self, version="unknown"):
            self.version = version

    def get_distribution(pkg_name):
        return MockDistribution("1.0.0")

    shim = types.ModuleType('pkg_resources')
    shim.resource_string = resource_string
    shim.resource_filename = resource_filename
    shim.get_distribution = get_distribution
    shim.require = lambda *args: []
    sys.modules['pkg_resources'] = shim
    logger.warning(
        "pkg_resources not importable — compatibility shim installed via importlib."
    )


# Install shim at module load time so it is ready before any vietocr imports.
_patch_pkg_resources()

def order_quad_points(points: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 điểm đa giác theo thứ tự: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def perspective_crop(image_bgr: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Cắt phối cảnh 4 điểm thành ảnh chữ nhật phẳng chuẩn."""
    ordered = order_quad_points(polygon)
    tl, tr, br, bl = ordered
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = max(int(max(width_a, width_b)), 1)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = max(int(max(height_a, height_b)), 1)

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(
        image_bgr,
        matrix,
        (max_w, max_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped


def expand_polygon(
    polygon: np.ndarray,
    img_h: int,
    img_w: int,
    pad_h_pct: float = 0.22,
    pad_w_pct: float = 0.04,
) -> np.ndarray:
    """Mở rộng đa giác 4 điểm trực tiếp trên ảnh gốc theo tỉ lệ chiều cao và ngang."""
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    center = np.mean(pts, axis=0)
    expanded = pts.copy()
    for i in range(len(expanded)):
        vec = expanded[i] - center
        expanded[i, 0] += vec[0] * pad_w_pct
        expanded[i, 1] += vec[1] * pad_h_pct
    expanded[:, 0] = np.clip(expanded[:, 0], 0, img_w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, img_h - 1)
    return expanded


def crop_with_adaptive_padding(
    image_bgr: np.ndarray,
    polygon: np.ndarray,
    pad_h_pct: float = 0.22,
    pad_w_pct: float = 0.04,
) -> np.ndarray:
    """
    Cắt phối cảnh trực tiếp từ ảnh gốc với tọa độ đa giác được mở rộng tự nhiên.
    Giúp lấy trọn vẹn toàn bộ dấu thanh tiếng Việt (mũ, hỏi, ngã, nặng, râu) cùng màu nền thực,
    ngăn ngừa triệt để hiện tượng biến dạng dấu (ví dụ: bị lặp biên khiến ^ thành dấu sắc ').
    """
    h_img, w_img = image_bgr.shape[:2]
    expanded_poly = expand_polygon(polygon, h_img, w_img, pad_h_pct=pad_h_pct, pad_w_pct=pad_w_pct)
    crop = perspective_crop(image_bgr, expanded_poly)
    return crop


def enhance_crop_contrast(crop_bgr: np.ndarray) -> np.ndarray:
    """Cân bằng độ tương phản CLAHE và làm nét nhẹ cho các vùng chữ mờ/chìm nền."""
    if crop_bgr.size == 0 or crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
        return crop_bgr

    try:
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Unsharp mask nhẹ
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.25, gaussian, -0.25, 0)
        return sharpened
    except Exception:
        return crop_bgr


def upscale_if_small(crop_bgr: np.ndarray, target_height: int = 64) -> np.ndarray:
    """Upscale ảnh crop nhỏ để tăng độ nét cho nhận dạng OCR."""
    h, w = crop_bgr.shape[:2]
    if h >= target_height or h <= 0 or w <= 0:
        return crop_bgr
    scale = target_height / float(h)
    new_w = max(int(w * scale), 1)
    return cv2.resize(crop_bgr, (new_w, target_height), interpolation=cv2.INTER_CUBIC)


def preserve_case_replace(text: str, pattern: str, replacement: str) -> str:
    """Thay thế pattern bằng replacement và bảo toàn phong cách viết hoa/thường."""
    def _repl(match):
        orig = match.group(0)
        if orig.isupper():
            return replacement.upper()
        if orig.istitle():
            return replacement.title()
        if orig.islower():
            return replacement.lower()
        return replacement

    return re.sub(pattern, _repl, text, flags=re.IGNORECASE)


def normalize_vietnamese_text(text: str) -> str:
    """
    Hậu xử lý chuẩn hóa Tiếng Việt & Sửa lỗi chính tả OCR phổ biến:
    1. Chuẩn hóa Unicode dựng sẵn NFC.
    2. Tách các cụm từ viết hoa bị dính liền thường gặp (CÀPHÊ -> CÀ PHÊ).
    3. Tự động sửa các lỗi nhầm dấu thanh kinh điển trong OCR tiếng Việt:
       - 'cà phé' -> 'cà phê'
       - 'phố cô' / 'phổ cổ' -> 'phố cổ'
       - 'hương vị truyền thông' -> 'hương vị truyền thống'
    4. Chuẩn hóa khoảng trắng quanh dấu câu.
    """
    if not text:
        return ""

    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text).strip()

    # 2. Sửa lỗi nhầm dấu và tách dính từ trong các cụm từ tiếng Việt phổ biến
    corrections = [
        # Nhầm dấu 'phé' thay vì 'phê'
        (r'\bcà\s*ph[éèẻẽẹ]\b', 'cà phê'),
        (r'\bcàph[êéèẻẽẹ]\b', 'cà phê'),
        # Dính từ phổ biến
        (r'\bviệtnam\b', 'việt nam'),
        (r'\bthờisự\b', 'thời sự'),
        (r'\btinnóng\b', 'tin nóng'),
        (r'\bhànội\b', 'hà nội'),
        (r'\bsàigòn\b', 'sài gòn'),
        (r'\bđànẵng\b', 'đà nẵng'),
        (r'\btphcm\b', 'tp.hcm'),
        # Ngữ cảnh phố cổ
        (r'\bphổ\s+cổ\b', 'phố cổ'),
        (r'\bphố\s+cô\b', 'phố cổ'),
        (r'\bphốcổ\b', 'phố cổ'),
        # Ngữ cảnh truyền thống
        (r'\b(hương\s+vị\s+)truyền\s+thông\b', r'\1truyền thống'),
        (r'\b(bản\s+sắc\s+)truyền\s+thông\b', r'\1truyền thống'),
    ]

    for pat, rep in corrections:
        if r'\1' in rep:
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        else:
            text = preserve_case_replace(text, pat, rep)

    # 3. Chuẩn hóa khoảng trắng quanh dấu câu (loại bỏ space trước dấu, bảo đảm space sau dấu)
    text = re.sub(r'\s+([,.:;?!])', r'\1', text)
    text = re.sub(r'([,.:;?!])(?=[^\s\d,.:;?!])', r'\1 ', text)

    # 4. Gom khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class ONNXDBNetDetector:
    """Bộ phát hiện vùng chữ DBNet siêu tốc chạy trên ONNX Runtime GPU (CUDA)."""

    def __init__(self, model_path: Optional[Path] = None, device_id: int = 0):
        self.model_path = model_path or DEFAULT_ONNX_PATH
        self.device_id = device_id
        self._ensure_model_exists()

        providers = [
            ("CUDAExecutionProvider", {"device_id": self.device_id}),
            "CPUExecutionProvider",
        ]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=providers)
        self.active_providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        logger.info(f"ONNX DBNet Detector loaded with providers: {self.active_providers}")

        self._warmup()

    def _ensure_model_exists(self):
        # Kiểm tra file có tồn tại và kích thước phải > 1MB (để tránh trường hợp file pointer Git LFS chưa được pull)
        if not self.model_path.exists() or self.model_path.stat().st_size < 1_000_000:
            logger.info(f"Downloading PP-OCRv4 detection ONNX model to {self.model_path}...")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="deepghs/paddleocr",
                filename="det/ch_PP-OCRv4_det/model.onnx",
                local_dir=str(BASE_DIR / "models"),
            )

    def _warmup(self):
        try:
            dummy = np.zeros((1, 3, 64, 64), dtype=np.float32)
            self.session.run(None, {self.input_name: dummy})
            logger.info("ONNX Detector warmup complete.")
        except Exception as e:
            logger.warning(f"Warmup warning: {e}")

    def preprocess(self, image_bgr: np.ndarray, max_side_len: int = 960) -> Tuple[np.ndarray, float, float]:
        h, w = image_bgr.shape[:2]
        ratio = 1.0
        if max(h, w) > max_side_len:
            ratio = max_side_len / float(max(h, w))
            resize_h = int(h * ratio)
            resize_w = int(w * ratio)
        else:
            resize_h = h
            resize_w = w

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        resized = cv2.resize(image_bgr, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (rgb - mean) / std
        chw = np.transpose(norm, (2, 0, 1))
        tensor = np.expand_dims(chw, axis=0).astype(np.float32)
        return tensor, ratio_h, ratio_w

    @staticmethod
    def _unclip(box: np.ndarray, unclip_ratio: float = 1.5) -> Optional[np.ndarray]:
        poly = Polygon(box)
        if poly.length == 0:
            return None
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        if not expanded:
            return None
        return np.array(expanded[0])

    def postprocess(
        self,
        pred_map: np.ndarray,
        ratio_h: float,
        ratio_w: float,
        orig_h: int,
        orig_w: int,
        thresh: float = 0.25,
        box_thresh: Optional[float] = None,
        unclip_ratio: float = 1.5,
    ) -> Tuple[List[np.ndarray], List[float]]:
        # box_thresh filters individual bounding boxes by their mean prediction score.
        # Default: thresh * 0.6 so it's always below the pixel-level mask threshold
        # and works correctly on both CPU and CUDA ONNX runtimes.
        if box_thresh is None:
            box_thresh = thresh * 0.6
        pred = pred_map[0, 0, :, :]
        mask = (pred > thresh).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes, scores = [], []

        for contour in contours:
            if len(contour) < 4:
                continue
            rect = cv2.minAreaRect(contour)
            box = np.array(cv2.boxPoints(rect), dtype=np.float32)

            box_int = np.int32(box)
            mask_box = np.zeros_like(pred, dtype=np.uint8)
            cv2.fillPoly(mask_box, [box_int], 1)
            mean_score = cv2.mean(pred, mask=mask_box)[0]

            if mean_score < box_thresh:
                continue

            expanded = self._unclip(box, unclip_ratio)
            if expanded is None:
                continue

            box_exp = cv2.boxPoints(cv2.minAreaRect(expanded))
            box_exp[:, 0] = np.clip(box_exp[:, 0] / ratio_w, 0, orig_w)
            box_exp[:, 1] = np.clip(box_exp[:, 1] / ratio_h, 0, orig_h)

            boxes.append(box_exp)
            scores.append(float(mean_score))

        return boxes, scores

    def detect(self, image_bgr: np.ndarray, thresh: float = 0.25) -> Tuple[List[np.ndarray], List[float]]:
        orig_h, orig_w = image_bgr.shape[:2]
        tensor, ratio_h, ratio_w = self.preprocess(image_bgr)
        preds = self.session.run(None, {self.input_name: tensor})[0]
        # Pass thresh so box_thresh is auto-derived as thresh*0.6 (consistent across CPU/CUDA)
        boxes, scores = self.postprocess(preds, ratio_h, ratio_w, orig_h, orig_w, thresh=thresh)
        return boxes, scores


class OCREngine:
    """Pipeline OCR cao cấp: ONNX DBNet (RTX 4060) + VietOCR GPU cải tiến."""

    def __init__(
        self,
        vietocr_model_name: str = "vgg_transformer",
        gpu_id: int = 0,
        det_thresh: float = 0.25,
        min_rec_confidence: float = 0.20,
    ):
        self.gpu_id = gpu_id
        self.det_thresh = det_thresh
        self.min_rec_confidence = min_rec_confidence
        self.vietocr_model_name = vietocr_model_name
        self.use_beamsearch = False
        self.device_name = "CPU"
        self.detector = None
        self.vietocr_predictor = None

        self._init_models()

    def _init_models(self):
        logger.info("Initializing GPU-Accelerated OCR Pipeline...")

        if torch.cuda.is_available():
            self.device_name = torch.cuda.get_device_name(self.gpu_id)
            self.torch_device = f"cuda:{self.gpu_id}"
            torch.cuda.set_device(self.gpu_id)
            torch.backends.cudnn.benchmark = True
            logger.info(f"PyTorch CUDA active on: {self.device_name}")
        else:
            self.device_name = "CPU"
            self.torch_device = "cpu"
            logger.info("Running on CPU.")

        # ONNX DBNet Detector
        try:
            self.detector = ONNXDBNetDetector(device_id=self.gpu_id)
        except Exception as e:
            logger.error(f"Error initializing ONNX Detector: {e}")
            self.detector = None

        # VietOCR Recognizer
        self._load_vietocr(self.vietocr_model_name, use_beamsearch=self.use_beamsearch)

    def _load_vietocr(self, model_name: str, use_beamsearch: bool = False):
        # Try loading on the primary device first, then fall back to CPU.
        # Note: _patch_pkg_resources() was already called at module load time,
        # so pkg_resources (or its shim) is available before any vietocr import.
        devices_to_try = [self.torch_device]
        if self.torch_device != "cpu":
            devices_to_try.append("cpu")

        last_error = None
        for device in devices_to_try:
            try:
                from vietocr.tool.config import Cfg
                from vietocr.tool.predictor import Predictor

                # Always create a fresh config for each device attempt
                config = Cfg.load_config_from_name(model_name)
                config["device"] = device
                config["cnn"]["pretrained"] = False
                config["predictor"]["beamsearch"] = use_beamsearch

                self.vietocr_predictor = Predictor(config)
                self.vietocr_model_name = model_name
                self.use_beamsearch = use_beamsearch
                logger.info(f"VietOCR [{model_name}] (beamsearch={use_beamsearch}) loaded on {device}.")

                # Warmup
                try:
                    dummy_crop = np.zeros((32, 128, 3), dtype=np.uint8)
                    self._fast_batch_predict([dummy_crop])
                    logger.info("VietOCR warmup complete.")
                except Exception as w_err:
                    logger.warning(f"VietOCR warmup skipped: {w_err}")
                return  # success

            except Exception as e:
                last_error = e
                logger.error(f"Failed to load VietOCR on {device}: {e}")
                if device != "cpu":
                    logger.info("Trying VietOCR on CPU fallback...")

        logger.error(f"VietOCR could not be loaded on any device. Last error: {last_error}")
        self.vietocr_predictor = None

    def switch_model(self, model_name: str, use_beamsearch: bool = False):
        if model_name != self.vietocr_model_name or use_beamsearch != self.use_beamsearch:
            self._load_vietocr(model_name, use_beamsearch=use_beamsearch)

    def _fast_batch_predict(
        self,
        crops_bgr: List[np.ndarray],
        target_h: int = 32,
        max_w_limit: int = 512,
    ) -> Tuple[List[str], List[float]]:
        """
        True Single-Pass GPU Batch Inference:
        Chuẩn hóa và gom tất cả các crops vào 1 Tensor duy nhất (N, 3, 32, max_W)
        chạy duy nhất 1 lần forward pass trên GPU thay vì lặp tuần tự từng ảnh.
        """
        from vietocr.tool.translate import translate

        tensors = []
        max_w = 32

        for crop in crops_bgr:
            h, w = crop.shape[:2]
            new_w = int(w * target_h / max(h, 1))
            new_w = max(32, min(new_w, max_w_limit))

            resized = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            tensors.append(chw)
            if new_w > max_w:
                max_w = new_w

        # Căn chỉnh max_w theo bội số của 8 để tối ưu hóa Tensor Core trên RTX 4060
        max_w = int(np.ceil(max_w / 8.0) * 8)

        batch = np.zeros((len(crops_bgr), 3, target_h, max_w), dtype=np.float32)
        for i, t in enumerate(tensors):
            w_cur = t.shape[2]
            batch[i, :, :, :w_cur] = t

        device = self.vietocr_predictor.device
        batch_tensor = torch.from_numpy(batch).to(device)

        with torch.inference_mode():
            translated, probs = translate(batch_tensor, self.vietocr_predictor.model)

        sents = self.vietocr_predictor.vocab.batch_decode(translated.tolist())
        probs_list = [float(p) for p in probs.tolist()]
        return sents, probs_list

    def recognize_crops(
        self,
        crops_bgr: List[np.ndarray],
        upscale_small: bool = True,
        contrast_boost: bool = False,
        normalize_text: bool = True,
    ) -> List[Tuple[str, float]]:
        """Nhận diện chữ từng ảnh crop bằng VietOCR theo batch GPU với tiền xử lý và chuẩn hóa tiếng Việt."""
        if not crops_bgr:
            return []
        if self.vietocr_predictor is None:
            logger.info("VietOCR predictor is None. Attempting reload...")
            self._load_vietocr(self.vietocr_model_name, use_beamsearch=self.use_beamsearch)
        if self.vietocr_predictor is None:
            logger.error("VietOCR predictor remains unavailable. Returning empty recognition.")
            return []

        # 1. Tiền xử lý tương phản & kích thước
        processed_crops = []
        for crop in crops_bgr:
            c = crop
            if contrast_boost:
                c = enhance_crop_contrast(c)
            if upscale_small and c.shape[0] < 42:
                c = upscale_if_small(c, 64)
            processed_crops.append(c)

        texts, probs = [], []

        # 2. Ưu tiên True Single-Pass Batch GPU Inference siêu tốc (khi không bật beamsearch)
        if not self.use_beamsearch:
            try:
                texts, probs = self._fast_batch_predict(processed_crops)
            except Exception as e:
                logger.warning(f"Fast batch prediction warning: {e}. Falling back to default predictor.")
                texts, probs = [], []

        # 3. Fallback hoặc chế độ Beam Search
        if not texts:
            pil_images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in processed_crops]
            try:
                with torch.inference_mode():
                    texts, probs = self.vietocr_predictor.predict_batch(
                        pil_images,
                        return_prob=True,
                    )
            except Exception as e:
                logger.warning(f"Batch prediction warning: {e}. Falling back to single-image mode.")
                texts, probs = [], []
                for img in pil_images:
                    try:
                        with torch.inference_mode():
                            t, p = self.vietocr_predictor.predict(img, return_prob=True)
                            texts.append(t)
                            probs.append(p)
                    except Exception:
                        texts.append("")
                        probs.append(0.0)

        # 4. Chuẩn hóa tiếng Việt & Sửa lỗi chính tả
        results = []
        for text, prob in zip(texts, probs):
            p_val = float(np.mean(prob)) if hasattr(prob, "__iter__") else float(prob or 0.0)
            cleaned = normalize_vietnamese_text(str(text)) if normalize_text else str(text).strip()
            results.append((cleaned, p_val))

        return results

    def predict(
        self,
        image_input: Any,
        det_thresh: Optional[float] = None,
        min_confidence: Optional[float] = None,
        vietocr_model: Optional[str] = None,
        upscale_small: bool = True,
        adaptive_padding: bool = True,
        contrast_boost: bool = False,
        normalize_text: bool = True,
        use_beamsearch: bool = False,
    ) -> Dict[str, Any]:
        """
        Toàn bộ luồng OCR GPU nâng cao:
        ONNX Detect -> Adaptive Crop -> Contrast Boost -> VietOCR Batch (Greedy/Beam) -> Normalize NFC.
        """
        start_time = time.perf_counter()

        # Update model config if changed
        target_model = vietocr_model or self.vietocr_model_name
        if target_model != self.vietocr_model_name or use_beamsearch != self.use_beamsearch:
            self.switch_model(target_model, use_beamsearch=use_beamsearch)

        thresh = det_thresh if det_thresh is not None else self.det_thresh
        min_conf = min_confidence if min_confidence is not None else self.min_rec_confidence

        # Decode image
        if isinstance(image_input, dict) and "path" in image_input:
            image_input = image_input["path"]

        if isinstance(image_input, (str, Path)):
            with open(str(image_input), "rb") as f:
                nparr = np.frombuffer(f.read(), np.uint8)
            image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            image_bgr = image_input.copy()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if image_bgr is None:
            raise ValueError("Failed to decode or read image.")

        orig_h, orig_w = image_bgr.shape[:2]

        # 1. Detect text regions (ONNX GPU)
        t_det_start = time.perf_counter()
        if self.detector is not None:
            polygons, det_scores = self.detector.detect(image_bgr, thresh=thresh)
        else:
            polygons, det_scores = [], []
        det_ms = round((time.perf_counter() - t_det_start) * 1000, 1)

        # 2. Crop với Adaptive Padding bảo vệ dấu tiếng Việt
        crops = []
        valid_indices = []
        for i, poly in enumerate(polygons):
            if adaptive_padding:
                crop = crop_with_adaptive_padding(image_bgr, poly, pad_h_pct=0.22, pad_w_pct=0.04)
            else:
                crop = perspective_crop(image_bgr, poly)

            if crop.shape[0] >= 6 and crop.shape[1] >= 10:
                crops.append(crop)
                valid_indices.append(i)

        filtered_polygons = [polygons[i] for i in valid_indices]
        filtered_det_scores = [det_scores[i] for i in valid_indices]

        # 3. Recognize với VietOCR (GPU)
        t_rec_start = time.perf_counter()
        rec_results = self.recognize_crops(
            crops,
            upscale_small=upscale_small,
            contrast_boost=contrast_boost,
            normalize_text=normalize_text,
        )
        rec_ms = round((time.perf_counter() - t_rec_start) * 1000, 1)

        # 4. Format & Filter by confidence
        detected_boxes = []
        for poly, det_sc, (text, rec_conf) in zip(filtered_polygons, filtered_det_scores, rec_results):
            if not text or rec_conf < min_conf:
                continue

            poly_pts = poly.reshape(-1, 2).tolist()
            center_y = float(poly.reshape(-1, 2)[:, 1].mean())
            min_x = float(poly.reshape(-1, 2)[:, 0].min())

            detected_boxes.append({
                "bbox": poly_pts,
                "text": text,
                "det_confidence": round(det_sc, 3),
                "rec_confidence": round(rec_conf, 3),
                "center_y": center_y,
                "min_x": min_x,
            })

        # 5. Natural reading order sort: top-to-bottom, left-to-right
        detected_boxes.sort(key=lambda b: (b["center_y"], b["min_x"]))

        for idx, b in enumerate(detected_boxes):
            b["index"] = idx + 1
            del b["center_y"]
            del b["min_x"]

        # Combined text
        full_text = "\n".join(b["text"] for b in detected_boxes)

        # 6. Draw annotated image with Unicode badges
        annotated_bgr = self.draw_bboxes(image_bgr, detected_boxes)

        # 7. Base64 encoding
        annotated_b64 = self._bgr_to_base64_url(annotated_bgr)
        original_b64 = self._bgr_to_base64_url(image_bgr)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

        return {
            "boxes": detected_boxes,
            "full_text": full_text,
            "image_width": orig_w,
            "image_height": orig_h,
            "elapsed_ms": elapsed_ms,
            "det_ms": det_ms,
            "rec_ms": rec_ms,
            "device": self.device_name,
            "providers": self.detector.active_providers if self.detector else [],
            "beamsearch": self.use_beamsearch,
            "annotated_image_base64": annotated_b64,
            "original_image_base64": original_b64,
        }

    @staticmethod
    def _get_font(size: int = 13) -> Any:
        font_candidates = [
            # Windows
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\tahoma.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            # Linux / Docker (Debian, Ubuntu, Hugging Face Spaces)
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
        for path in font_candidates:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    def draw_bboxes(self, image_bgr: np.ndarray, boxes: List[Dict[str, Any]]) -> np.ndarray:
        """Vẽ bounding box đa giác dạ quang và nhãn thứ tự lên ảnh hỗ trợ tiếng Việt Unicode."""
        canvas = image_bgr.copy()
        overlay = image_bgr.copy()

        neon_cyan = (248, 189, 56)  # BGR

        for b in boxes:
            pts = np.asarray(b["bbox"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], neon_cyan)

        # Subtle translucent fill
        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)

        # Draw borders with OpenCV
        for b in boxes:
            pts = np.asarray(b["bbox"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=True, color=neon_cyan, thickness=2, lineType=cv2.LINE_AA)

        # Convert to PIL Image for crisp Vietnamese Unicode text badges
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = self._get_font(13)

        for b in boxes:
            pts = np.asarray(b["bbox"], dtype=np.int32).reshape(-1, 2)
            min_x = int(np.min(pts[:, 0]))
            min_y = int(np.min(pts[:, 1]))
            badge_text = f"#{b['index']} {b['text'][:25]}" + ("..." if len(b['text']) > 25 else "")

            bbox = font.getbbox(badge_text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            pad_x, pad_y = 6, 4
            badge_y1 = max(0, min_y - text_h - pad_y * 2 - 4)
            badge_y2 = badge_y1 + text_h + pad_y * 2
            badge_x1 = max(0, min_x)
            badge_x2 = min(pil_img.width, badge_x1 + text_w + pad_x * 2)

            # Rounded badge background
            draw.rounded_rectangle(
                [badge_x1, badge_y1, badge_x2, badge_y2],
                radius=4,
                fill=(15, 18, 26),
                outline=(56, 189, 248),
                width=1,
            )
            # Unicode Vietnamese text rendering
            draw.text(
                (badge_x1 + pad_x, badge_y1 + pad_y),
                badge_text,
                font=font,
                fill=(255, 255, 255),
            )

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _bgr_to_base64_url(img_bgr: np.ndarray, quality: int = 90) -> str:
        success, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return ""
        b64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
