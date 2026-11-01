import { env, AutoModel, AutoProcessor, RawImage } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2";

const els = {
  status: document.getElementById("model-status"),
  statusTitle: document.getElementById("model-status-title"),
  statusDetail: document.getElementById("model-status-detail"),
  progressFill: document.getElementById("progress-fill"),
  dropSection: document.getElementById("drop-section"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  resultSection: document.getElementById("result-section"),
  engineBadge: document.getElementById("engine-badge"),
  compare: document.getElementById("compare"),
  imgBefore: document.getElementById("img-before"),
  afterWrap: document.getElementById("after-wrap"),
  canvasAfter: document.getElementById("canvas-after"),
  slider: document.getElementById("slider"),
  sliderLine: document.querySelector(".slider-line"),
  download: document.getElementById("download"),
  newPhoto: document.getElementById("new-photo"),
  bgColor: document.getElementById("bg-color"),
};

env.allowLocalModels = false;

const MODEL_ID = "Xenova/modnet";
let model = null;
let processor = null;
let engine = "";
let matte = null;      // Uint8Array mask, image size
let srcImage = null;   // RawImage
let objectUrl = null;

function setProgress(fraction) {
  els.progressFill.style.width = Math.round(fraction * 100) + "%";
}

async function webgpuAvailable() {
  try {
    if (!navigator.gpu) return false;
    return !!(await navigator.gpu.requestAdapter());
  } catch (_) { return false; }
}

async function loadModel() {
  const progress_callback = (p) => {
    if (p.status === "progress" && p.total) setProgress(p.loaded / p.total);
    else if (p.status === "done") setProgress(1);
  };
  const opts = [];
  if (await webgpuAvailable()) {
    opts.push({ device: "webgpu", dtype: "q8", progress_callback, label: "WebGPU (GPU accelerated)" });
  }
  opts.push({ device: "wasm", dtype: "q8", progress_callback, label: "CPU (WASM)" });
  let lastErr = null;
  for (const o of opts) {
    try {
      const { label, ...loadOpts } = o;
      model = await AutoModel.from_pretrained(MODEL_ID, loadOpts);
      processor = await AutoProcessor.from_pretrained(MODEL_ID);
      engine = label;
      return;
    } catch (e) { lastErr = e; model = null; }
  }
  throw lastErr || new Error("Model failed to load");
}

function showError(msg) {
  els.status.classList.add("error");
  els.status.querySelector(".spinner").style.display = "none";
  els.statusTitle.textContent = "Could not load the AI model";
  els.statusDetail.textContent = msg;
}

async function init() {
  try {
    await loadModel();
    els.status.classList.add("hidden");
    els.dropSection.classList.remove("hidden");
  } catch (e) {
    showError("This browser could not run the model (" + (e && e.message ? e.message : e) + "). Try current Chrome or Edge on desktop.");
  }
}

async function processFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  els.dropSection.classList.add("hidden");
  els.status.classList.remove("hidden");
  els.statusTitle.textContent = "Removing the background";
  els.statusDetail.textContent = "Usually a few seconds on desktop, longer on phones.";
  setProgress(0.5);

  try {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    const img = await RawImage.fromURL(objectUrl);
    const { pixel_values } = await processor(img);
    let output;
    try {
      ({ output } = await model({ input: pixel_values }));
    } catch (inferErr) {
      if (engine.startsWith("WebGPU")) {
        // GPU inference failed at runtime - fall back to CPU and retry once
        model = await AutoModel.from_pretrained(MODEL_ID, { device: "wasm", dtype: "q8" });
        engine = "CPU (WASM)";
        ({ output } = await model({ input: pixel_values }));
      } else {
        throw inferErr;
      }
    }
    const maskData = (await RawImage.fromTensor(output[0].mul(255).to("uint8")).resize(img.width, img.height)).data;

    srcImage = img;
    matte = maskData;
    els.imgBefore.src = objectUrl;
    els.imgBefore.onload = () => {
      els.compare.style.aspectRatio = img.width + " / " + img.height;
    };
    renderResult();
    els.engineBadge.textContent = engine;
    els.status.classList.add("hidden");
    els.resultSection.classList.remove("hidden");
    els.resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    showError("Could not process that image (" + (e && e.message ? e.message : e) + "). Try a smaller JPG or PNG.");
    setTimeout(() => {
      els.status.classList.add("hidden");
      els.dropSection.classList.remove("hidden");
    }, 4000);
  }
}

function currentBg() {
  const checked = document.querySelector('input[name="bg"]:checked').value;
  if (checked === "white") return "#ffffff";
  if (checked === "color") return els.bgColor.value;
  return null; // transparent
}

function renderResult() {
  if (!srcImage || !matte) return;
  const { width, height } = srcImage;
  const canvas = els.canvasAfter;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");

  const bg = currentBg();
  if (bg) { ctx.fillStyle = bg; ctx.fillRect(0, 0, width, height); }

  ctx.drawImage(srcImage.toCanvas(), 0, 0);
  const pixels = ctx.getImageData(0, 0, width, height);
  for (let i = 0; i < matte.length; ++i) {
    const a = matte[i];
    if (bg) {
      // composite over bg: out = fg*a + bg*(1-a)
      const t = a / 255;
      const bi = 4 * i;
      const bgc = parseInt(bg.slice(1), 16);
      pixels.data[bi]     = pixels.data[bi] * t     + ((bgc >> 16) & 255) * (1 - t);
      pixels.data[bi + 1] = pixels.data[bi + 1] * t + ((bgc >> 8) & 255) * (1 - t);
      pixels.data[bi + 2] = pixels.data[bi + 2] * t + (bgc & 255) * (1 - t);
      pixels.data[bi + 3] = 255;
    } else {
      pixels.data[4 * i + 3] = a;
    }
  }
  ctx.putImageData(pixels, 0, 0);
  updateSlider();
}

function updateSlider() {
  const pct = els.slider.value;
  els.afterWrap.style.clipPath = "inset(0 0 0 " + pct + "%)";
  els.imgBefore.style.clipPath = "inset(0 " + (100 - pct) + "% 0 0)";
  els.sliderLine.style.left = pct + "%";
}

// --- events ---
els.dropzone.addEventListener("click", () => els.fileInput.click());
els.dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.fileInput.click(); } });
els.fileInput.addEventListener("change", () => processFile(els.fileInput.files[0]));
["dragenter", "dragover"].forEach((ev) => els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((ev) => els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove("dragover"); }));
els.dropzone.addEventListener("drop", (e) => { const f = e.dataTransfer.files && e.dataTransfer.files[0]; if (f) processFile(f); });
window.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData ? e.clipboardData.items : [])].find((i) => i.type.startsWith("image/"));
  if (item && model) processFile(item.getAsFile());
});
els.slider.addEventListener("input", updateSlider);
document.querySelectorAll('input[name="bg"]').forEach((r) => r.addEventListener("change", renderResult));
els.bgColor.addEventListener("input", () => {
  document.querySelector('input[name="bg"][value="color"]').checked = true;
  renderResult();
});
els.download.addEventListener("click", () => {
  els.canvasAfter.toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "background-removed.png";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }, "image/png");
});
els.newPhoto.addEventListener("click", () => {
  els.resultSection.classList.add("hidden");
  els.dropSection.classList.remove("hidden");
  els.fileInput.value = "";
  matte = null;
  srcImage = null;
});

init();
