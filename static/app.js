document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const processingBanner = document.getElementById('processing-banner');
  const resultsDashboard = document.getElementById('results-dashboard');
  
  const systemStatus = document.getElementById('system-status');
  const statusText = document.getElementById('status-text');
  const statusDot = systemStatus.querySelector('.status-dot');
  
  const settingsToggle = document.getElementById('settings-toggle');
  const settingsDrawer = document.getElementById('settings-drawer');
  const closeSettings = document.getElementById('close-settings');
  
  const detThresh = document.getElementById('det-thresh');
  const detThreshVal = document.getElementById('det-thresh-val');
  const minConfidence = document.getElementById('min-confidence');
  const minConfidenceVal = document.getElementById('min-confidence-val');
  const vietocrModel = document.getElementById('vietocr-model');
  const upscaleSmall = document.getElementById('upscale-small');
  const adaptivePadding = document.getElementById('adaptive-padding');
  const normalizeText = document.getElementById('normalize-text');
  const contrastBoost = document.getElementById('contrast-boost');
  const useBeamsearch = document.getElementById('use-beamsearch');
  
  const annotatedImg = document.getElementById('annotated-img');
  const originalImg = document.getElementById('original-img');
  const bboxCanvas = document.getElementById('bbox-overlay-canvas');
  const imageWrapper = document.getElementById('image-wrapper');
  
  const tabBtns = document.querySelectorAll('.tab-btn');
  const btnZoomIn = document.getElementById('btn-zoom-in');
  const btnZoomOut = document.getElementById('btn-zoom-out');
  const btnZoomReset = document.getElementById('btn-zoom-reset');
  const btnDownloadImg = document.getElementById('btn-download-img');
  
  const statCount = document.getElementById('stat-count');
  const statTime = document.getElementById('stat-time');
  const statConf = document.getElementById('stat-conf');
  const imgResolution = document.getElementById('img-resolution');
  const boxCountBadge = document.getElementById('box-count-badge');
  
  const fullTextDisplay = document.getElementById('full-text-display');
  const boxesList = document.getElementById('boxes-list');
  const btnCopyAll = document.getElementById('btn-copy-all');
  const btnExportTxt = document.getElementById('btn-export-txt');
  const btnExportJson = document.getElementById('btn-export-json');
  const sampleChips = document.querySelectorAll('.sample-chip');
  const toast = document.getElementById('toast');

  // State
  let currentZoom = 1.0;
  let currentOCRData = null;
  let highlightedIndex = null;

  // Check API Health
  async function checkHealth() {
    try {
      const response = await fetch('/api/health');
      if (response.ok) {
        const data = await response.json();
        statusDot.className = 'status-dot ready';
        const isCuda = (data.providers || []).some(p => p.includes('CUDA'));
        statusText.textContent = `${data.device || 'GPU'} ${isCuda ? '(ONNX + CUDA)' : ''} sẵn sàng`;
      } else {
        statusText.textContent = 'Server bận';
      }
    } catch (err) {
      statusText.textContent = 'Chưa kết nối server';
    }
  }
  checkHealth();
  setInterval(checkHealth, 20000);

  // Settings Events
  settingsToggle.addEventListener('click', () => settingsDrawer.classList.toggle('open'));
  closeSettings.addEventListener('click', () => settingsDrawer.classList.remove('open'));
  
  detThresh.addEventListener('input', (e) => detThreshVal.textContent = e.target.value);
  minConfidence.addEventListener('input', (e) => minConfidenceVal.textContent = e.target.value);

  // File Upload Handlers
  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) processFile(e.target.files[0]);
  });

  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  });

  // Clipboard Paste Support (Ctrl + V)
  window.addEventListener('paste', (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (const item of items) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const blob = item.getAsFile();
        showToast('Đã nhận ảnh dán từ Clipboard!');
        processFile(blob);
        break;
      }
    }
  });

  // Sample Images Trigger
  sampleChips.forEach(chip => {
    chip.addEventListener('click', async () => {
      const sampleId = chip.getAttribute('data-sample');
      try {
        setLoading(true);
        const params = new URLSearchParams({
          det_thresh: detThresh.value,
          min_confidence: minConfidence.value,
          model_name: vietocrModel.value,
          upscale_small: upscaleSmall.checked,
          adaptive_padding: adaptivePadding ? adaptivePadding.checked : true,
          normalize_text: normalizeText ? normalizeText.checked : true,
          contrast_boost: contrastBoost ? contrastBoost.checked : false,
          use_beamsearch: useBeamsearch ? useBeamsearch.checked : false
        });
        const res = await fetch(`/api/sample/${sampleId}?${params.toString()}`);
        if (!res.ok) throw new Error('Không tải được ảnh mẫu');
        const data = await res.json();
        renderResults(data);
      } catch (err) {
        showToast('Lỗi tải mẫu: ' + err.message);
      } finally {
        setLoading(false);
      }
    });
  });

  // Process File with Backend API
  async function processFile(file) {
    if (!file.type.startsWith('image/')) {
      showToast('Vui lòng chọn file hình ảnh hợp lệ');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('det_thresh', detThresh.value);
    formData.append('min_confidence', minConfidence.value);
    formData.append('model_name', vietocrModel.value);
    formData.append('upscale_small', upscaleSmall.checked);
    formData.append('adaptive_padding', adaptivePadding ? adaptivePadding.checked : true);
    formData.append('normalize_text', normalizeText ? normalizeText.checked : true);
    formData.append('contrast_boost', contrastBoost ? contrastBoost.checked : false);
    formData.append('use_beamsearch', useBeamsearch ? useBeamsearch.checked : false);

    try {
      const response = await fetch('/api/ocr', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Lỗi nhận diện OCR từ server');
      }

      const data = await response.json();
      renderResults(data);
      showToast('Đã trích xuất văn bản thành công!');
    } catch (err) {
      showToast(`Lỗi: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  function setLoading(loading) {
    if (loading) {
      processingBanner.style.display = 'flex';
      resultsDashboard.style.display = 'none';
      processingBanner.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } else {
      processingBanner.style.display = 'none';
    }
  }

  // Render Result UI
  function renderResults(data) {
    currentOCRData = data;
    resultsDashboard.style.display = 'grid';

    // Update Images
    annotatedImg.src = data.annotated_image_base64;
    originalImg.src = data.original_image_base64 || data.annotated_image_base64;
    btnDownloadImg.href = data.annotated_image_base64;

    // Reset zoom
    resetZoom();

    // Stats
    statCount.textContent = data.boxes.length;
    statTime.textContent = `${data.elapsed_ms || 0} ms`;
    if (data.det_ms && data.rec_ms) {
      statTime.title = `Phát hiện (Det): ${data.det_ms}ms | Nhận diện (Rec): ${data.rec_ms}ms`;
    }
    
    const avgConf = data.boxes.length > 0
      ? (data.boxes.reduce((sum, b) => sum + (b.rec_confidence || 0), 0) / data.boxes.length * 100).toFixed(1)
      : 0;
    statConf.textContent = `${avgConf}%`;

    boxCountBadge.textContent = `${data.boxes.length} vùng chữ`;
    if (data.image_width && data.image_height) {
      imgResolution.textContent = `Kích thước: ${data.image_width} × ${data.image_height}px`;
    }

    // Full Text
    fullTextDisplay.textContent = data.full_text || '(Không phát hiện thấy chữ)';

    // BBox List
    boxesList.innerHTML = '';
    if (data.boxes.length === 0) {
      boxesList.innerHTML = '<div style="color: var(--text-dim); text-align: center; padding: 20px;">Không tìm thấy văn bản phù hợp</div>';
    } else {
      data.boxes.forEach((box, idx) => {
        const boxItem = document.createElement('div');
        boxItem.className = 'box-item';
        boxItem.dataset.index = idx;

        const confPct = Math.round((box.rec_confidence || 0) * 100);
        boxItem.innerHTML = `
          <div class="box-left">
            <span class="box-index">#${idx + 1}</span>
            <div class="box-text">${escapeHtml(box.text)}</div>
          </div>
          <div class="box-right">
            <span class="confidence-tag">${confPct}%</span>
            <div class="confidence-bar">
              <div class="confidence-fill" style="width: ${confPct}%"></div>
            </div>
          </div>
        `;

        // Hover events to highlight on canvas
        boxItem.addEventListener('mouseenter', () => {
          highlightBox(idx);
          boxItem.classList.add('highlighted');
        });

        boxItem.addEventListener('mouseleave', () => {
          highlightBox(null);
          boxItem.classList.remove('highlighted');
        });

        // Click to copy item text
        boxItem.addEventListener('click', () => {
          navigator.clipboard.writeText(box.text);
          showToast(`Đã sao chép: "${box.text}"`);
        });

        boxesList.appendChild(boxItem);
      });
    }

    // Setup Canvas Overlay
    setupCanvas();

    resultsDashboard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // Setup and draw Canvas Overlay
  function setupCanvas() {
    annotatedImg.onload = () => {
      syncCanvasDimensions();
      drawBBoxOverlay();
    };
    if (annotatedImg.complete) {
      syncCanvasDimensions();
      drawBBoxOverlay();
    }
  }

  function syncCanvasDimensions() {
    bboxCanvas.width = annotatedImg.naturalWidth;
    bboxCanvas.height = annotatedImg.naturalHeight;
  }

  function drawBBoxOverlay() {
    const ctx = bboxCanvas.getContext('2d');
    ctx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);

    if (!currentOCRData || !currentOCRData.boxes) return;

    currentOCRData.boxes.forEach((box, idx) => {
      if (!box.bbox || box.bbox.length === 0) return;
      
      const isHighlighted = (highlightedIndex === idx);
      const pts = box.bbox;

      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pts[i][0], pts[i][1]);
      }
      ctx.closePath();

      if (isHighlighted) {
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 4;
        ctx.fillStyle = 'rgba(56, 189, 248, 0.3)';
        ctx.shadowColor = '#38bdf8';
        ctx.shadowBlur = 15;
        ctx.fill();
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    });
  }

  function highlightBox(index) {
    highlightedIndex = index;
    drawBBoxOverlay();
  }

  // View Tabs
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const view = btn.dataset.view;

      if (view === 'annotated') {
        annotatedImg.style.display = 'block';
        originalImg.style.display = 'none';
        bboxCanvas.style.display = 'block';
      } else {
        annotatedImg.style.display = 'none';
        originalImg.style.display = 'block';
        bboxCanvas.style.display = 'none';
      }
    });
  });

  // Zoom Controls
  btnZoomIn.addEventListener('click', () => {
    currentZoom = Math.min(currentZoom + 0.25, 3.0);
    applyZoom();
  });

  btnZoomOut.addEventListener('click', () => {
    currentZoom = Math.max(currentZoom - 0.25, 0.5);
    applyZoom();
  });

  btnZoomReset.addEventListener('click', resetZoom);

  function resetZoom() {
    currentZoom = 1.0;
    applyZoom();
  }

  function applyZoom() {
    imageWrapper.style.transform = `scale(${currentZoom})`;
    btnZoomReset.textContent = `${Math.round(currentZoom * 100)}%`;
  }

  // Copy & Export Actions
  btnCopyAll.addEventListener('click', () => {
    if (!currentOCRData || !currentOCRData.full_text) {
      showToast('Không có văn bản để sao chép');
      return;
    }
    navigator.clipboard.writeText(currentOCRData.full_text);
    showToast('Đã sao chép toàn bộ văn bản vào bộ nhớ tạm!');
  });

  btnExportTxt.addEventListener('click', () => {
    if (!currentOCRData || !currentOCRData.full_text) return;
    downloadFile(currentOCRData.full_text, 'ocr_result.txt', 'text/plain;charset=utf-8');
    showToast('Đã tải xuống file .TXT');
  });

  btnExportJson.addEventListener('click', () => {
    if (!currentOCRData) return;
    const jsonStr = JSON.stringify(currentOCRData, null, 2);
    downloadFile(jsonStr, 'ocr_result.json', 'application/json;charset=utf-8');
    showToast('Đã tải xuống file .JSON');
  });

  function downloadFile(content, fileName, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 3000);
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
});
