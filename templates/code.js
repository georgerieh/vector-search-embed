import {
  InferenceSession,
  Tensor,
  env,
} from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.min.mjs";
import {
  FaceDetector,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs";

// Source - https://stackoverflow.com/a/23522755
// Posted by fregante, modified by community. See post 'Timeline' for change history
// Retrieved 2026-08-23, License - CC BY-SA 3.0

var isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

if (isSafari) document.getElementById('photo').parentElement.style.display = 'none';

const loadingContainer = document.getElementById("loading-container");
const loadingBar = document.getElementById("loading-bar");
const loadingText = document.getElementById("loading-text");
if (loadingContainer) loadingContainer.style.display = "block";

loadingText.textContent = "Checking model cache...";
loadingBar.style.width = "0%";
const btn = document.querySelector('button[name="forwardBtn"]');
btn.disabled = true;
window.gridSelected = window.gridSelected || new Set();
const gridSelected = window.gridSelected;
const CACHE_NAME = "ai-models-v1";

async function fetchWithCacheAndProgress(url, onProgress) {
    const cache = await caches.open(CACHE_NAME);

    const cachedResponse = await cache.match(url);

    // 1. Return immediately if model is already in browser cache
    if (cachedResponse) {
      onProgress(1.0);
      return await cachedResponse.arrayBuffer();
    }

  // 2. Otherwise download and stream
  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok)
    throw new Error(`Failed to download ${url}: ${response.statusText}`);

  // Clone response to store in Cache API for future loads
  cache.put(url, response.clone());

  const contentLength = response.headers.get("Content-Length");
  const total = contentLength ? parseInt(contentLength, 10) : null;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    if (total) onProgress(received / total);
  }

  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return buffer.buffer;
}

loadingText.textContent = "Checking model cache...";
loadingBar.style.width = "0%";

let progressDino = 0;
let progressFacenet = 0;

function updateCombinedProgress() {
  const combined = progressDino * 0.85 + progressFacenet * 0.15;
  const display = Math.round(combined * 85);
  loadingBar.style.width = display + "%";
  loadingText.textContent = `Loading AI Models... ${display}%`;
}

if (!isSafari) {
const [dinoBuffer, facenetBuffer, vision] = await Promise.all([
  fetchWithCacheAndProgress(
    "https://huggingface.co/georgerieh/onnx-dino-vitb-16-and-facenet/resolve/main/dino_vitb16_inline.onnx",
    (pct) => {
      progressDino = pct;
      updateCombinedProgress();
    },
  ),
  fetchWithCacheAndProgress(
    "https://huggingface.co/georgerieh/onnx-dino-vitb-16-and-facenet/resolve/main/facenet_inline.onnx",
    (pct) => {
      progressFacenet = pct;
      updateCombinedProgress();
    },
  ),
  FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm",
  ),
]);


const [dinoSession, faceNetSession, faceDetector] = await Promise.all([
  InferenceSession.create(dinoBuffer, {
    executionProviders: ["webgpu", "wasm"],
  }),
  InferenceSession.create(facenetBuffer, {
    executionProviders: ["webgpu", "wasm"],
  }),
  FaceDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    },
    runningMode: "IMAGE",
    minDetectionConfidence: 0.9,
  }),
]);
};

loadingText.textContent = "Final steps...";
loadingBar.style.width = "85%";


loadingBar.style.width = "100%";
loadingText.textContent = "✓ Ready — upload a photo to search";
btn.disabled = false;

setTimeout(() => {
  loadingBar.style.width = "0%";
  loadingText.textContent = "";
  document.getElementById("loading-container").style.display = "none";
}, 1500);

function preprocessDino(imgElement) {
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgElement, 0, 0, 224, 224);
  const { data } = ctx.getImageData(0, 0, 224, 224);
  const tensor = new Float32Array(3 * 224 * 224);
  for (let i = 0; i < 224 * 224; i++) {
    tensor[i] = (data[i * 4] / 255.0 - 0.5) / 0.5;
    tensor[i + 224 * 224] = (data[i * 4 + 1] / 255.0 - 0.5) / 0.5;
    tensor[i + 2 * 224 * 224] = (data[i * 4 + 2] / 255.0 - 0.5) / 0.5;
  }
  return new Tensor("float32", tensor, [1, 3, 224, 224]);
}

function preprocessFace(imgElement, box) {
  const canvas = document.createElement("canvas");
  canvas.width = 160;
  canvas.height = 160;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(
    imgElement,
    box.originX,
    box.originY,
    box.width,
    box.height,
    0,
    0,
    160,
    160,
  );
  const { data } = ctx.getImageData(0, 0, 160, 160);
  const tensor = new Float32Array(3 * 160 * 160);
  for (let i = 0; i < 160 * 160; i++) {
    tensor[i] = (data[i * 4] / 255.0 - 0.5) / 0.5;
    tensor[i + 160 * 160] = (data[i * 4 + 1] / 255.0 - 0.5) / 0.5;
    tensor[i + 2 * 160 * 160] = (data[i * 4 + 2] / 255.0 - 0.5) / 0.5;
  }
  return new Tensor("float32", tensor, [1, 3, 160, 160]);
}

function normalize(arr) {
  const norm = Math.sqrt(arr.reduce((s, v) => s + v * v, 0));
  return arr.map((v) => v / norm);
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = async () => {
      try {
        await img.decode();
        resolve(img);
      } catch (err) {
        console.error("Image decode failed:", err);
        resolve(img);
      }
    };
    img.onerror = (err) => reject(err);
    img.src = URL.createObjectURL(file);
  });
}
window.leafletMap = null;
let markerClusters = null;

function initMap() {
  if (leafletMap) return;
  leafletMap = L.map("map").setView([48.505, 2.33], 3);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "© OpenStreetMap",
  }).addTo(leafletMap);
  leafletMap.on("zoomend moveend", updateHexLayer);
  updateHexLayer();
}
let hexLayer = null;

function updateHexLayer() {
  if (!leafletMap) return;
  const zoom = leafletMap.getZoom();
  // map zoom to h3 resolution
  const res =
    zoom < 1
      ? 1
      : zoom < 2
        ? 2
        : zoom < 4
          ? 3
          : zoom < 6
            ? 4
            : zoom < 8
              ? 5
              : zoom < 10
                ? 6
                : 7;

  fetch(`/hex_coverage?resolution=${res}`)
    .then((r) => r.json())
    .then((data) => {
      if (hexLayer) leafletMap.removeLayer(hexLayer);
      const layers = [];
      data.cells.forEach((cell) => {
        const boundary = h3
          .cellToBoundary(cell.h3)
          .map(([lat, lng]) => [lat, lng]);
        const poly = L.polygon(boundary, {
          color: "#007aff",
          fillColor: "#007aff",
          fillOpacity: 0.08,
          weight: 1,
          opacity: 0.4,
        });
        poly.bindTooltip(`${cell.count} photos`, { sticky: true });
        poly.on("click", () => {
          const h3id = cell.h3;
          try {
            navigator.clipboard.writeText(h3id);
          } catch {
            // fallback for non-https
            const ta = document.createElement("textarea");
            ta.value = h3id;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
          }
          // also put in h3 filter input
          document.getElementById("h3-filter").value = h3id;
          poly.setStyle({ fillOpacity: 0.25 });
          poly.bindTooltip(`Copied: ${h3id}`, { sticky: true }).openTooltip();
          setTimeout(() => poly.setStyle({ fillOpacity: 0.08 }), 600);
        });
        layers.push(poly);
      });
      hexLayer = L.layerGroup(layers).addTo(leafletMap);
    });
}
initMap();
document
  .querySelector('form[name="input"]')
  .addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("image-in");
    const startDate = document.getElementById("start-date").value;
    const endDate = document.getElementById("end-date").value;
    const limit = document.getElementById("limit").value || 50;
    const country = document.getElementById("country-filter").value;
    const city = document.getElementById("city-filter").value;
    const h3cell = document.getElementById("h3-filter").value;

    const hasImage = fileInput.files.length > 0;
    const hasFilters = startDate || endDate || country || city || h3cell;

    if (!hasImage && !hasFilters && !selectedFaceEmbedding) return;

    document.getElementById("loading-container").style.display = "block";
    window.currentImages = [];
    document.getElementById("photo-grid").innerHTML = "";
    clearGridSelection();

    let embedding = null,
      facenetEmbedding = null;

    if (hasImage) {
      loadingBar.style.width = "0%";
      btn.disabled = true;
      const img = await loadImage(fileInput.files[0]);
      loadingText.textContent = "Computing visual embedding...";
      loadingBar.style.width = "30%";
      const dinoInputName = dinoSession.inputNames[0];
      const dinoFeeds = {};
      dinoFeeds[dinoInputName] = preprocessDino(img);

      const dinoResults = await dinoSession.run(dinoFeeds);
      const dinoOutputName = dinoSession.outputNames[0];
      embedding = normalize(
        Array.from(dinoResults[dinoOutputName].data).slice(0, 768),
      );
      loadingBar.style.width = "60%";
      if (document.getElementById("detect-faces").checked) {
        console.log("Preparing face detection framework...");
        loadingText.textContent = "Detecting faces...";

        let detection = { detections: [] };

        try {
          if (!faceDetector) {
            throw new Error(
              "MediaPipe FaceDetector instance is not initialized.",
            );
          }

          console.log("Passing image to MediaPipe WASM runtime...", img);
          detection = faceDetector.detect(img);
          console.log(
            "MediaPipe completed successfully. Found faces:",
            detection.detections.length,
          );
        } catch (e) {
          console.error("Face detection step crashed or timed out:", e);
          detection = { detections: [] };
        }

        if (detection.detections && detection.detections.length > 0) {
          loadingText.textContent = "Recognizing face...";
          loadingBar.style.width = "80%";

          const best = detection.detections.reduce((a, b) =>
            a.categories[0].score > b.categories[0].score ? a : b,
          );

          const facenetInputName = faceNetSession.inputNames[0];
          const facenetFeeds = {};
          facenetFeeds[facenetInputName] = preprocessFace(
            img,
            best.boundingBox,
          );

          const faceResults = await faceNetSession.run(facenetFeeds);
          const facenetOutputName = faceNetSession.outputNames[0];
          facenetEmbedding = normalize(
            Array.from(faceResults[facenetOutputName].data),
          );

          const cropCanvas = document.createElement("canvas");
          cropCanvas.width = 160;
          cropCanvas.height = 160;
          const cropCtx = cropCanvas.getContext("2d");
          const box = best.boundingBox;
          cropCtx.drawImage(
            img,
            box.originX,
            box.originY,
            box.width,
            box.height,
            0,
            0,
            160,
            160,
          );

          const isKnown = savedFaces.some((f) => {
            const dot = f.embedding.reduce(
              (s, v, i) => s + v * facenetEmbedding[i],
              0,
            );
            return dot > 0.7;
          });
          if (!isKnown) {
            showFaceModal(facenetEmbedding, cropCanvas, 10000);
          }
        }
      }
      URL.revokeObjectURL(img.src);
      btn.disabled = false;
    } else {
      loadingBar.style.width = "50%";
      loadingText.textContent = "Searching...";
    }

    const response = await fetch("/search_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        embedding,
        facenet_embedding: facenetEmbedding || selectedFaceEmbedding,
        start_date: startDate,
        end_date: endDate,
        limit,
        country,
        city,
        h3cell,
      }),
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line) continue;
        const chunk = JSON.parse(line);
        window.renderChunk(chunk.images);
      }
    }
    loadingBar.style.width = "100%";
    loadingText.textContent = `${window.currentImages.length} results`;
    setTimeout(() => {
      document.getElementById("loading-container").style.display = "none";
    }, 600);
    setView("grid");
  });
document
  .getElementById("btn-open-face-upload")
  .addEventListener("click", () => {
    document.getElementById("face-upload-modal").style.display = "flex";
  });

document.getElementById("face-upload-close").addEventListener("click", () => {
  document.getElementById("face-upload-modal").style.display = "none";
});

const dropZone = document.getElementById("face-drop-zone");
const faceFileInput = document.getElementById("face-file-input");

dropZone.addEventListener("click", () => faceFileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "#007aff";
  dropZone.style.background = "#f0f7ff";
});

dropZone.addEventListener("dragleave", () => {
  dropZone.style.borderColor = "#c7c7cc";
  dropZone.style.background = "#fafafa";
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "#c7c7cc";
  dropZone.style.background = "#fafafa";
  if (e.dataTransfer.files.length) {
    processFaceImage(e.dataTransfer.files[0]);
  }
});

faceFileInput.addEventListener("change", (e) => {
  if (e.target.files.length) {
    processFaceImage(e.target.files[0]);
  }
});
function cleanPhotoUrl(url) {
  if (!url) return "";

  let clean = decodeURIComponent(url).replace(/^\//, "");

  const prefixesToRemove = [
    "files/",
    "thumbnail/",
    "static/",
    "media/georgerieh/T7/photos_from_icloud/",
    "media/georgerieh/T7/",
  ];

  let changed = true;
  while (changed) {
    changed = false;
    for (const prefix of prefixesToRemove) {
      if (clean.startsWith(prefix)) {
        clean = clean.slice(prefix.length).replace(/^\//, "");
        changed = true;
      }
    }
  }

  return clean;
}
async function processFaceImage(file) {
  const loadingText = document.getElementById("face-upload-loading");
  loadingText.style.display = "block";
  dropZone.style.display = "none";

  try {
    const img = await loadImage(file);

    let detection = { detections: [] };
    if (faceDetector) {
      detection = faceDetector.detect(img);
    }

    if (detection.detections && detection.detections.length > 0) {
      // Find the most confident face
      const best = detection.detections.reduce((a, b) =>
        a.categories[0].score > b.categories[0].score ? a : b,
      );

      // Run FaceNet
      const facenetInputName = faceNetSession.inputNames[0];
      const facenetFeeds = {};
      facenetFeeds[facenetInputName] = preprocessFace(img, best.boundingBox);

      const faceResults = await faceNetSession.run(facenetFeeds);
      const facenetOutputName = faceNetSession.outputNames[0];
      const facenetEmbedding = normalize(
        Array.from(faceResults[facenetOutputName].data),
      );

      // Crop it for the thumbnail
      const cropCanvas = document.createElement("canvas");
      cropCanvas.width = 160;
      cropCanvas.height = 160;
      const cropCtx = cropCanvas.getContext("2d");
      const box = best.boundingBox;
      cropCtx.drawImage(
        img,
        box.originX,
        box.originY,
        box.width,
        box.height,
        0,
        0,
        160,
        160,
      );

      // Hide upload modal and trigger your existing save face modal
      document.getElementById("face-upload-modal").style.display = "none";
      showFaceModal(facenetEmbedding, cropCanvas, 60000); // Allow 60s to type a name
    } else {
      alert("No face detected in this image. Try a clearer photo.");
    }
  } catch (err) {
    console.error("Face processing failed:", err);
    alert("An error occurred while processing the face.");
  } finally {
    // Reset UI
    loadingText.style.display = "none";
    dropZone.style.display = "flex";
    faceFileInput.value = "";
  }
}
let _favSet = null;

async function loadFavoritesFromDisk() {
  const resp = await fetch("/favorites");
  const { photos } = await resp.json();
  _favSet = new Set(photos.map((p) => cleanPhotoUrl(p.url)));
  const stored = JSON.parse(localStorage.getItem("mem_favorites") || "[]");
  const unsaved = stored
    .map((u) => cleanPhotoUrl(u))
    .filter((u) => !_favSet.has(u));

  if (unsaved.length > 0) {
    const save = confirm(
      `You have ${unsaved.length} favorite(s) saved only in browser storage. Save them to disk permanently?`,
    );
    if (save) {
      for (const url of unsaved) {
        await fetch("/favorites/add", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: url }),
        });
        _favSet.add(url);
      }
      localStorage.removeItem("mem_favorites");
    }
  }
  renderFavStrip();
  return _favSet;
}

function getFavorites() {
  return _favSet ? [..._favSet] : [];
}

async function toggleFavorite(url) {
  const cleanUrl = cleanPhotoUrl(url);

  if (!_favSet) await loadFavoritesFromDisk();

  if (_favSet.has(cleanUrl)) {
    _favSet.delete(cleanUrl);
    await fetch("/favorites/remove", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: cleanUrl }),
    });
  } else {
    _favSet.add(cleanUrl);
    await fetch("/favorites/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: cleanUrl }),
    });
  }
  renderFavStrip();
  return _favSet.has(cleanUrl);
}

// ── LIGHTBOX ──
{
  let lightboxImages = [],
    lightboxIndex = 0;
  let to_preload_others = true;
  function openLightbox(images, index, multiple = to_preload_others) {
    lightboxImages = images;
    lightboxIndex = index;
    let to_preload_others = multiple;
    updateLightbox();
    document.getElementById("lightbox").classList.add("open");
    document.body.style.overflow = "hidden";
  }
  function closeLightbox() {
    document.getElementById("lightbox").classList.remove("open");
    document.body.style.overflow = "";
  }
  const BASE_PATH = "media/georgerieh/T7/photos_from_icloud/";
  const BASE = BASE_PATH;

  function updateLightbox() {
    const img = lightboxImages[lightboxIndex];
    const url = img.url.replace(/^\//, "");
    const BASE = "media/georgerieh/T7/photos_from_icloud/";

    const cleanUrl = url.startsWith(BASE) ? url.slice(BASE.length) : url;
    const lightboxImg = document.getElementById("lightbox-img");
    lightboxImg.src = `/files/${url}`;
    lightboxImg.style.filter = "blur(2px)";
    const full = new Image();
    full.onload = () => {
      lightboxImg.src = full.src;
      lightboxImg.style.filter = "";
    };
    full.src = `/files/${url}`;
    document.getElementById("lightbox-location").textContent =
      "Date: " + img.date + " at " + (img.city + ", " + img.country) || "";
    document.getElementById("lightbox-counter").textContent =
      `${lightboxIndex + 1} / ${lightboxImages.length}`;
    document
      .getElementById("lightbox-fav")
      .classList.toggle("active", _favSet ? _favSet.has(cleanUrl) : false);
    const preload = (i) => {
      const p = lightboxImages[i];
      if (!p) return;
      const l = new Image();
      l.src = `/files/${p.url.replace(/^\//, "")}`;
    };
    preload((lightboxIndex + 1) % lightboxImages.length);
    preload(
      (lightboxIndex - 1 + lightboxImages.length) % lightboxImages.length,
    );
  }

  document
    .getElementById("lightbox-close")
    .addEventListener("click", closeLightbox);
  document.getElementById("lightbox-prev").addEventListener("click", () => {
    if (to_preload_others) {
      lightboxIndex =
        (lightboxIndex - 1 + lightboxImages.length) % lightboxImages.length;
      updateLightbox();
    }
  });
  document.getElementById("lightbox-next").addEventListener("click", () => {
    if (to_preload_others) {
      lightboxIndex = (lightboxIndex + 1) % lightboxImages.length;
      updateLightbox();
    }
  });
  document.getElementById("lightbox-fav").addEventListener("click", () => {
    const url = lightboxImages[lightboxIndex].url.replace(/^\//, "");
    document
      .getElementById("lightbox-fav")
      .classList.toggle("active", toggleFavorite(url));
  });
  document.getElementById("lightbox").addEventListener("click", (e) => {
    if (e.target === document.getElementById("lightbox")) closeLightbox();
  });
  document.addEventListener("keydown", (e) => {
    if (!document.getElementById("lightbox").classList.contains("open")) return;
    if (e.key === "Escape") closeLightbox();
    if (e.key === "ArrowLeft") {
      lightboxIndex =
        (lightboxIndex - 1 + lightboxImages.length) % lightboxImages.length;
      updateLightbox();
    }
    if (e.key === "ArrowRight") {
      lightboxIndex = (lightboxIndex + 1) % lightboxImages.length;
      updateLightbox();
    }
  });

  window.openLightbox = openLightbox;
}
// ── MAP ──

function updateMap(images) {
  initMap();
  leafletMap.on("zoomend moveend", updateHexLayer);
  updateHexLayer();
  const bounds = [];
  if (bounds.length > 0) leafletMap.fitBounds(bounds, { padding: [20, 20] });
}

// ── RENDER ──
window.currentImages = [];
/*document.addEventListener('DOMContentLoaded', () => {
    const interval = setInterval(() => {
        if (window.initMap) {
            window.initMap();
            clearInterval(interval);
        }
    }, 100);
    })*/
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const thumb = entry.target;
        const url = thumb.dataset.thumb;
        if (url && !thumb.style.backgroundImage) {
          thumb.style.backgroundImage = `url('${url}')`;
        }
        observer.unobserve(thumb);
      }
    });
  },
  { rootMargin: "200px", threshold: 0 },
);
window.renderChunk = function (images) {
  const grid = document.getElementById("photo-grid");
  const favs = _favSet ? [..._favSet] : [];
  const offset = window.currentImages.length;
  window.currentImages.push(...images);
  document.getElementById("empty-state").style.display = "none";
  images.forEach((img, i) => {
    const url = img.url.replace(/^\//, "");
    const isFav = favs.includes(url);
    const cell = document.createElement("div");
    cell.dataset.url = url;
    cell.className = "grid-cell";
    cell.draggable = true;
    cell.addEventListener("dragstart", (e) => {
      // if multiple selected, drag all of them
      const urls = gridSelected.size > 0 ? [...gridSelected] : [url];
      e.dataTransfer.setData("photo-paths", JSON.stringify(urls));
      e.dataTransfer.effectAllowed = "link";
      // visual feedback
      cell.style.opacity = "0.5";
    });
    cell.addEventListener("dragend", () => {
      cell.style.opacity = "";
    });
    cell.innerHTML = `
            <div class="grid-thumb" data-thumb="/thumbnail/${url}">
                <div class="grid-overlay">
                    <button class="thumb-btn thumb-fav ${isFav ? "fav-active" : ""}" title="Favorite">♥</button>
                    <button class="thumb-btn thumb-delete" title="Delete">✕</button>
                </div>
            </div>
        `;
    cell.querySelector(".grid-thumb").addEventListener("click", (e) => {
      if (e.target.closest(".thumb-btn")) return;
      if (gridSelected.size > 0 || e.shiftKey) {
        if (gridSelected.has(url)) {
          gridSelected.delete(url);
          cell.classList.remove("grid-selected");
        } else {
          gridSelected.add(url);
          cell.classList.add("grid-selected");
        }
        updateGridToolbar();
        return;
      }
      openLightbox(window.currentImages, offset + i, false);
      cell.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        gridSelected.add(url);
        cell.classList.add("grid-selected");
        updateGridToolbar();
      });
    });
    cell.querySelector(".thumb-fav").addEventListener("click", (e) => {
      e.stopPropagation();
      const isFav = toggleFavorite(url);
      e.target.classList.toggle("fav-active", isFav);
    });
    cell.querySelector(".thumb-delete").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm("Delete this photo?")) return;
      const fd = new FormData();
      fd.append("image_paths", url);
      await fetch("/delete_photo", { method: "POST", body: fd });
      cell.style.opacity = "0";
      cell.style.transform = "scale(0.85)";
      setTimeout(() => cell.remove(), 300);
    });
    grid.appendChild(cell);
    requestAnimationFrame(() => {
      cell.style.transitionDelay = `${Math.min(i * 15, 300)}ms`;
      cell.classList.add("appear");
    });
  });

  // lazy load new thumbs
  grid.querySelectorAll(".grid-thumb:not([data-observed])").forEach((el) => {
    el.dataset.observed = "1";
    observer.observe(el);
  });
};
window.renderResults = function (data) {
  window.currentImages = data.images;
  const grid = document.getElementById("photo-grid");
  grid.innerHTML = "";
  const favs = getFavorites();

  const fragment = document.createDocumentFragment();

  data.images.forEach((img, i) => {
    const url = img.url.replace(/^\//, "");
    const thumbUrl = `/thumbnail/${url}`;
    const isFav = favs.includes(url);
    const isSelected = gridSelected && gridSelected.has(url);

    const cell = document.createElement("div");
    cell.dataset.url = url;
    cell.className = `grid-cell ${isSelected ? "grid-selected" : ""}`;

    cell.innerHTML = `
      <div class="grid-thumb">
          <div class="grid-overlay">
              <button class="thumb-btn thumb-fav ${isFav ? "fav-active" : ""}" title="Favorite">♥</button>
              <button class="thumb-btn thumb-delete" title="Delete">✕</button>
          </div>
      </div>
    `;

    const thumbEl = cell.querySelector(".grid-thumb");
    thumbEl.style.backgroundImage = `url(${JSON.stringify(thumbUrl)})`;

    thumbEl.addEventListener("click", (e) => {
      if (e.target.closest(".thumb-btn")) return;

      if (e.shiftKey || e.ctrlKey || e.metaKey) {
        if (gridSelected.has(url)) {
          gridSelected.delete(url);
          cell.classList.remove("grid-selected");
        } else {
          gridSelected.add(url);
          cell.classList.add("grid-selected");
        }
        if (typeof updateGridToolbar === "function") updateGridToolbar();
        return;
      }

      if (typeof openLightbox === "function") {
        openLightbox(window.currentImages, i, false);
      } else {
        console.error("openLightbox function is not defined globally.");
      }
    });

    cell.querySelector(".thumb-fav").addEventListener("click", (e) => {
      e.stopPropagation();
      const nowFav = toggleFavorite(url);
      e.currentTarget.classList.toggle("fav-active", nowFav);
    });

    cell.querySelector(".thumb-delete").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm("Delete this photo?")) return;

      const fd = new FormData();
      fd.append("image_paths", url);

      try {
        const res = await fetch("/delete_photo", { method: "POST", body: fd });
        if (!res.ok) throw new Error("Deletion failed");

        // Sync state memory
        window.currentImages = window.currentImages.filter(
          (item) => item.url.replace(/^\//, "") !== url,
        );
        gridSelected.delete(url);
        if (typeof updateGridToolbar === "function") updateGridToolbar();

        // Animate and clean up DOM node
        cell.style.opacity = "0";
        cell.style.transform = "scale(0.85)";
        setTimeout(() => cell.remove(), 300);
      } catch (err) {
        alert("Could not delete photo.");
      }
    });

    fragment.appendChild(cell);
  });

  // Single DOM append operation for performance
  grid.appendChild(fragment);

  // Staggered appear animation
  requestAnimationFrame(() => {
    grid.querySelectorAll(".grid-cell").forEach((el, i) => {
      el.style.transitionDelay = `${Math.min(i * 25, 600)}ms`;
      el.classList.add("appear");
    });
  });

  if (typeof observer !== "undefined") {
    grid.querySelectorAll(".grid-thumb").forEach((el) => observer.observe(el));
  }

  const toolbarInfo = document.getElementById("toolbar-info");
  if (toolbarInfo) toolbarInfo.textContent = `${data.images.length} results`;

  if (typeof updateMap === "function") updateMap(data.images);

  const emptyState = document.getElementById("empty-state");
  if (emptyState) emptyState.style.display = "none";

  // If in gallery view, rebuild
  if (window.currentView === "gallery" && typeof buildGallery === "function") {
    buildGallery(data.images, 1);
  }

  if (typeof loadFavoritesFromDisk === "function") {
    loadFavoritesFromDisk();
  }
};

window.favOverlaySelected = new Set();
function renderFavStrip() {
  const strip = document.getElementById("fav-strip");
  strip.innerHTML = "";
  const favs = getFavorites();
  if (favs.length === 0) {
    strip.innerHTML =
      '<span style="font-size:12px;color:#8e8e93">No favorites yet</span>';
    return;
  }
  favs.forEach((url) => {
    const el = document.createElement("div");
    el.className = "fav-thumb";
    el.style.backgroundImage = `url('/thumbnail/${url}')`;
    el.title = url.split("/").pop();
    el.addEventListener("click", () => {
      const allFavs = getFavorites().map((u) => ({
        url: u,
        score: 0,
        lat: null,
        lon: null,
        location: null,
      }));
      const i = allFavs.findIndex((img) => img.url === url);
      openLightbox(allFavs, i >= 0 ? i : 0);
    });
    strip.appendChild(el);
  });
}
window.renderFavStrip = renderFavStrip;
async function openFavoritesOverlay() {
  const overlay = document.getElementById("favorites-overlay");
  const grid = document.getElementById("fav-overlay-grid");
  const countEl = document.getElementById("fav-overlay-count");
  const batchBtn = document.getElementById("fav-batch-delete-btn");

  overlay.style.display = "flex";
  grid.innerHTML = "";
  window.favOverlaySelected.clear();
  if (batchBtn) batchBtn.style.display = "none";

  const resp = await fetch("/favorites");
  const { photos } = await resp.json();
  countEl.textContent = `${photos.length} photos`;

  photos.forEach((photo, i) => {
    const url = photo.url.replace(/^\//, "");
    const cell = document.createElement("div");
    cell.className = "fav-overlay-cell";
    cell.dataset.url = url;
    cell.style.cssText =
      "aspect-ratio:1;position:relative;border-radius:4px;overflow:hidden;transition:transform 0.2s, box-shadow 0.2s;";

    cell.innerHTML = `
            <div class="fav-img-bg" style="width:100%;height:100%;background:url('/thumbnail/${url}') center/cover;cursor:pointer;"></div>
            <div class="grid-overlay" style="position:absolute;inset:0;background:linear-gradient(160deg,rgba(0,0,0,0.4) 0%,transparent 45%);opacity:0;transition:opacity 0.18s;padding:7px;display:flex;justify-content:space-between;align-items:flex-start;pointer-events:none;">
                <button class="thumb-btn thumb-fav fav-active" style="pointer-events:auto;" title="Unfavorite">♥</button>
                <button class="thumb-btn thumb-delete" style="pointer-events:auto;" title="Delete from disk">✕</button>
            </div>
            <div class="select-check" style="position:absolute;bottom:8px;right:8px;width:20px;height:20px;border-radius:50%;background:#007aff;color:white;display:none;align-items:center;justify-content:center;font-size:12px;font-weight:bold;box-shadow:0 2px 4px rgba(0,0,0,0.2);">✓</div>
        `;

    // Selection / Lightbox logic
    cell.querySelector(".fav-img-bg").addEventListener("click", (e) => {
      if (e.shiftKey || window.favOverlaySelected.size > 0) {
        // Toggle multi-select status
        if (window.favOverlaySelected.has(url)) {
          window.favOverlaySelected.delete(url);
          cell.style.transform = "";
          cell.style.boxShadow = "";
          cell.querySelector(".select-check").style.display = "none";
        } else {
          window.favOverlaySelected.add(url);
          cell.style.transform = "scale(0.92)";
          cell.style.boxShadow = "0 0 0 3px #007aff";
          cell.querySelector(".select-check").style.display = "flex";
        }

        // Show/Hide batch delete button
        if (batchBtn) {
          batchBtn.style.display =
            window.favOverlaySelected.size > 0 ? "inline-block" : "none";
          batchBtn.textContent = `✕ Delete Selected (${window.favOverlaySelected.size}) From Disk`;
        }
      } else {
        // Regular single click goes to Lightbox
        openLightbox(photos, i);
      }
    });

    // Hover overlays
    cell.addEventListener(
      "mouseenter",
      () => (cell.querySelector(".grid-overlay").style.opacity = "1"),
    );
    cell.addEventListener(
      "mouseleave",
      () => (cell.querySelector(".grid-overlay").style.opacity = "0"),
    );

    // Individual quick buttons inside the overlay
    cell.querySelector(".thumb-fav").addEventListener("click", async (e) => {
      e.stopPropagation();
      await toggleFavorite(url);
      cell.style.opacity = "0";
      setTimeout(() => {
        cell.remove();
        countEl.textContent = `${grid.children.length} photos`;
      }, 300);
    });

    cell.querySelector(".thumb-delete").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm("Delete this photo from disk permanently?")) return;
      const fd = new FormData();
      fd.append("image_paths", url);
      await fetch("/delete_photo", { method: "POST", body: fd });
      await toggleFavorite(url);
      cell.style.opacity = "0";
      setTimeout(() => {
        cell.remove();
        countEl.textContent = `${grid.children.length} photos`;
      }, 300);
    });

    grid.appendChild(cell);
  });
}
window.openFavoritesOverlay = openFavoritesOverlay;
async function deleteSelectedFavs() {
  if (window.favOverlaySelected.size === 0) return;

  const count = window.favOverlaySelected.size;
  if (
    !confirm(
      `Are you sure you want to permanently delete these ${count} selected photos from your T7 disk drive?`,
    )
  )
    return;

  const grid = document.getElementById("fav-overlay-grid");
  const countEl = document.getElementById("fav-overlay-count");
  const batchBtn = document.getElementById("fav-batch-delete-btn");

  document.getElementById("loading-container").style.display = "block";

  for (const url of [...window.favOverlaySelected]) {
    try {
      // 1. Purge physical asset from disk database
      const fd = new FormData();
      fd.append("image_paths", url);
      await fetch("/delete_photo", { method: "POST", body: fd });

      // 2. Clear out of virtual tracking registry file
      if (_favSet) _favSet.delete(url);
      await fetch("/favorites/remove", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: url }),
      });

      // 3. Drop from the visual overlay DOM tree
      const cell = grid.querySelector(`[data-url="${CSS.escape(url)}"]`);
      if (cell) {
        cell.style.opacity = "0";
        cell.style.transform = "scale(0.8)";
        setTimeout(() => cell.remove(), 250);
      }

      // Also clean out standard view grid tracking arrays if open behind the overlay
      window.currentImages = window.currentImages.filter(
        (img) => img.url.replace(/^\//, "") !== url,
      );
      const mainGridCell = document.querySelector(
        `#photo-grid [data-url="${CSS.escape(url)}"]`,
      );
      if (mainGridCell) mainGridCell.remove();
    } catch (err) {
      console.error("Batch deletion processing failure for item:", url, err);
    }
  }

  // Reset selections states cleanly
  window.favOverlaySelected.clear();
  if (batchBtn) batchBtn.style.display = "none";

  setTimeout(() => {
    countEl.textContent = `${grid.children.length} photos`;
    renderFavStrip();
    document.getElementById("loading-container").style.display = "none";
  }, 300);
}
window.deleteSelectedFavs = deleteSelectedFavs;

document.getElementById("fav-overlay-close").addEventListener("click", () => {
  document.getElementById("favorites-overlay").style.display = "none";
});
let savedFaces = [];
let selectedFaceEmbedding = null;

async function loadFaces() {
  const resp = await fetch("/faces");
  const { faces } = await resp.json();
  savedFaces = faces;
  renderFaceSelector();
}

function renderFaceSelector() {
  const container = document.getElementById("face-selector");
  const msg = document.getElementById("no-faces-msg");
  container.innerHTML = "";
  if (!savedFaces.length) {
    msg.style.display = "block";
    return;
  }
  msg.style.display = "none";
  savedFaces.forEach((face) => {
    const el = document.createElement("div");
    el.style.cssText = "position:relative;cursor:pointer;";
    el.innerHTML = `
                <img src="${face.thumbnail}" style="
                    width:44px;height:44px;border-radius:50%;object-fit:cover;
                    border:2px solid transparent;transition:border-color 0.15s;
                " title="${face.name}">
                <div style="font-size:9px;text-align:center;color:#8e8e93;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;width:44px;">${face.name}</div>
            `;
    const img = el.querySelector("img");
    el.addEventListener("click", () => {
      if (selectedFaceEmbedding === face.embedding) {
        selectedFaceEmbedding = null;
        img.style.borderColor = "transparent";
      } else {
        selectedFaceEmbedding = face.embedding;
        // deselect others
        container
          .querySelectorAll("img")
          .forEach((i) => (i.style.borderColor = "transparent"));
        img.style.borderColor = "#007aff";
      }
    });
    // long press to delete
    let pressTimer;
    el.addEventListener("mousedown", () => {
      pressTimer = setTimeout(async () => {
        if (!confirm(`Delete face "${face.name}"?`)) return;
        await fetch(`/faces/${face.id}`, { method: "DELETE" });
        loadFaces();
      }, 800);
    });
    el.addEventListener("mouseup", () => clearTimeout(pressTimer));
    container.appendChild(el);
  });
}

function showFaceModal(faceEmbedding, croppedCanvas, autoClose = 10000) {
  return new Promise((resolve) => {
    const timeout = setTimeout(() => {
      modal.style.display = "none";
      resolve();
    }, autoClose);

    const done = () => {
      clearTimeout(timeout);
      resolve();
    };
    const modal = document.getElementById("face-modal");
    const canvas = document.getElementById("face-crop-canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, 160, 160);
    ctx.drawImage(croppedCanvas, 0, 0, 160, 160);
    modal.style.display = "flex";

    document.getElementById("face-name-input").value = "";
    document.getElementById("face-name-input").focus();

    document.getElementById("face-modal-save").onclick = async () => {
      const name = document.getElementById("face-name-input").value.trim();
      if (!name) {
        document.getElementById("face-name-input").focus();
        return;
      }
      const thumbnail = canvas.toDataURL("image/jpeg", 0.8);
      await fetch("/faces", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, embedding: faceEmbedding, thumbnail }),
      });
      modal.style.display = "none";
      loadFaces();
      done();
    };
    document.getElementById("face-modal-cancel").onclick = () => {
      modal.style.display = "none";
      done();
    };
    document
      .getElementById("face-name-input")
      .addEventListener("keydown", (e) => {
        if (e.key === "Enter")
          document.getElementById("face-modal-save").click();
      });
  });
}

loadFaces();
document
  .getElementById("memory-today-btn")
  .addEventListener("click", async () => {
    document.getElementById("memory-loading").style.display = "flex";
    document.getElementById("memory-content").style.display = "none";

    const resp = await fetch("/photo_of_day");
    const data = await resp.json();
    currentMemory = data;

    document.getElementById("memory-title").textContent = data.title;
    document.getElementById("memory-subtitle").textContent = data.subtitle;

    if (data.photos.length > 0) {
      const heroUrl = data.photos[0].url.replace(/^\//, "");
      const hero = document.getElementById("memory-hero");
      hero.style.backgroundImage = `url('/thumbnail/${heroUrl}?size=800')`;
      hero.onclick = () =>
        window.openLightbox && window.openLightbox(data.photos, 0);
    }

    const grid = document.getElementById("memory-grid");
    grid.innerHTML = "";
    data.photos.slice(1).forEach((photo, i) => {
      const url = photo.url.replace(/^\//, "");
      const cell = document.createElement("div");
      cell.style.cssText = "aspect-ratio:1;border-radius:4px;cursor:pointer;";
      cell.style.background = `url('/thumbnail/${url}') center/cover`;
      cell.addEventListener(
        "click",
        () => window.openLightbox && window.openLightbox(data.photos, i + 1),
      );
      grid.appendChild(cell);
    });

    document.getElementById("memory-loading").style.display = "none";
    document.getElementById("memory-content").style.display = "block";
  });

fetch("/autocomplete")
  .then((r) => r.json())
  .then((data) => {
    const cl = document.getElementById("country-list");
    data.countries.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c;
      cl.appendChild(opt);
    });
    const ci = document.getElementById("city-list");
    data.cities.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c;
      ci.appendChild(opt);
    });
  });
const DOW_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

async function loadStats() {
  document.getElementById("info-pane").scrollTop = 0;
  document.getElementById("info-loading").style.display = "flex";
  document.getElementById("info-content").style.display = "none";
  const data =
    window._statsData || (await fetch("/stats").then((r) => r.json()));
  window._statsData = data;
  document.getElementById("info-loading").style.display = "none";
  document.getElementById("info-content").style.display = "block";
  document.getElementById("total-badge").textContent =
    data.total.toLocaleString() + " photos";
  renderMonthChart(data.by_month);
  renderDowChart(data.by_dow);
  renderMiniTable("top-cities-table", data.top_cities, "top");
  renderMiniTable("top-countries-table", data.top_countries, "top");
  renderMiniTable("bottom-cities-table", data.bottom_cities, "bottom");
  renderMiniTable("bottom-countries-table", data.bottom_countries, "bottom");
}

function renderMonthChart(byMonth) {
  const el = document.getElementById("month-chart");
  const valid = byMonth.filter((d) => d.month && d.month.includes(":"));
  if (!valid.length) {
    el.innerHTML = '<span style="color:#8e8e93;font-size:12px">No data</span>';
    return;
  }
  const max = Math.max(...valid.map((d) => d.count));
  valid.forEach((d) => {
    const [year, month] = d.month.split(":");
    const names = [
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec",
    ];
    const pct = Math.round((d.count / max) * 100);
    const label = names[parseInt(month) - 1] + " " + year.slice(2);
    const col = document.createElement("div");
    col.className = "bar-col";
    col.innerHTML = `<div class="bar" style="height:${Math.max(pct, 1)}%"><div class="bar-tooltip">${label}: ${d.count.toLocaleString()}</div></div><div class="bar-label">${label}</div>`;
    el.appendChild(col);
  });
}

function renderDowChart(byDow) {
  const el = document.getElementById("dow-chart");
  const valid = byDow.filter((d) => d.dow !== null && d.dow !== undefined);
  if (!valid.length) {
    el.innerHTML = '<span style="color:#8e8e93;font-size:12px">No data</span>';
    return;
  }
  const counts = Array(7).fill(0);
  valid.forEach((d) => {
    counts[d.dow] = d.count;
  });
  const max = Math.max(...counts);
  counts.forEach((count, dow) => {
    const pct = max > 0 ? Math.round((count / max) * 100) : 0;
    const row = document.createElement("div");
    row.className = "dow-row";
    row.innerHTML = `<div class="dow-label">${DOW_NAMES[dow]}</div><div class="dow-track"><div class="dow-bar" style="width:${pct}%"></div></div><div class="dow-count">${count.toLocaleString()}</div>`;
    el.appendChild(row);
  });
}

function renderMiniTable(id, rows, type) {
  const table = document.getElementById(id);
  if (!rows.length) {
    table.innerHTML =
      '<tr><td style="color:#8e8e93;font-size:12px">No data</td></tr>';
    return;
  }
  rows.forEach((row, i) => {
    const tr = document.createElement("tr");
    const pillClass = type === "top" ? `rank-${i + 1}` : "bottom-pill";
    tr.innerHTML = `<td><span class="rank-pill ${pillClass}">${i + 1}</span>${row.name}</td><td>${row.count.toLocaleString()}</td>`;
    table.appendChild(tr);
  });
}

// preload silently after 2s
setTimeout(() => {
  fetch("/stats")
    .then((r) => r.json())
    .then((data) => {
      window._statsData = data;
    })
    .catch(() => {});
}, 2000);

window.currentImages = [];
window.currentView = "grid";

window.setView = function (view) {
  window.currentView = view;

  ["grid", "gallery", "info", "memory", "review"].forEach((v) => {
    document.getElementById(`btn-${v}`)?.classList.toggle("active", v === view);
  });

  const hasResults = window.currentImages
    ? window.currentImages.length > 0
    : false;
  const fullWidthViews = ["gallery", "memory", "info", "review"];
  const isFullWidth = fullWidthViews.includes(view);

  document.getElementById("results-area").style.display =
    view === "grid" && hasResults ? "flex" : "none";

  document.getElementById("map-pane").style.display =
    view === "grid" && hasResults ? "flex" : "none";

  document.getElementById("empty-state").style.display =
    view === "grid" && !hasResults ? "flex" : "none";

  document
    .getElementById("gallery-pane")
    .classList.toggle("active", view === "gallery");

  document.getElementById("info-pane").style.display =
    view === "info" ? "block" : "none";

  document.getElementById("memory-pane").style.display =
    view === "memory" ? "flex" : "none";

  document.getElementById("review-pane").style.display =
    view === "review" ? "flex" : "none";

  document.querySelector(".sidebar").style.display = isFullWidth ? "none" : "";

  document.querySelector(".main").style.marginLeft = isFullWidth
    ? "0"
    : "320px";

  if (view === "gallery" && hasResults) {
    buildGallery(window.currentImages, 1);
  }

  if (view === "info") {
    if (!window.statsLoaded) {
      window.statsLoaded = true;
      loadStats();
    }
    document.getElementById("info-pane").scrollTop = 0;
  }

  if (view === "memory") {
    loadMemory();
  }

  if (
    view === "review" &&
    !document.getElementById("review-grid").children.length
  ) {
    loadReviewBatch();
  }
};
let galleryIndex = 1;

// Track active fetch requests so we can cancel them on jumps
const activeFetchControllers = new Map(); // index -> AbortController

function buildGallery(images, startIndex) {
  const main = document.getElementById("gallery-main");
  const strip = document.getElementById("gallery-strip");

  if (!main || !strip) return;

  cancelAllDownloads();
  main.querySelectorAll(".gallery-photo").forEach((el) => el.remove());
  strip.innerHTML = "";

  window.currentImages = images;
  galleryIndex = startIndex;

  const videoExtensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"];

  images.forEach((img, i) => {
    const url = img.url.replace(/^\//, "");
    const ext = url.substring(url.lastIndexOf(".")).toLowerCase();
    const isVideo =
      img.is_video || img.type === "video" || videoExtensions.includes(ext);

    const mainSrc = isVideo ? `/thumbnail/${url}` : `/files/${url}`;
    const thumbSrc = `/thumbnail/${url}`;

    const photo = document.createElement("div");
    photo.className = "gallery-photo" + (i === startIndex ? " active" : "");
    photo.dataset.src = mainSrc;
    photo.dataset.thumb = thumbSrc;
    photo.dataset.index = i;
    photo.style.cursor = "pointer";

    photo.addEventListener("click", () => {
      window.open(`/files/${url}`, "_blank").focus();
    });

    const controls = main.querySelector(".gallery-controls");
    if (controls) {
      main.insertBefore(photo, controls);
    } else {
      main.appendChild(photo);
    }

    const thumb = document.createElement("div");
    thumb.className = "strip-thumb" + (i === startIndex ? " active" : "");
    thumb.style.backgroundImage = `url('${thumbSrc}')`;
    thumb.addEventListener("click", () => setGalleryIndex(i));
    strip.appendChild(thumb);
  });

  preloadSmart(startIndex);
}

function cancelAllDownloads() {
  activeFetchControllers.forEach((controller) => controller.abort());
  activeFetchControllers.clear();
}

function preloadSmart(currentIndex) {
  const photos = document.querySelectorAll(".gallery-photo");
  if (!photos.length) return;

  // 1. Calculate ideal window: [-1, current, +1, +2, +3, +4, +5]
  const neededIndices = new Set([
    currentIndex,
    currentIndex + 1,
    currentIndex - 1,
    currentIndex + 2,
    currentIndex + 3,
    currentIndex + 4,
    currentIndex + 5,
  ]);

  // 2. CANCEL downloads for photos outside this new window (frees up bandwidth for skips)
  activeFetchControllers.forEach((controller, idx) => {
    if (!neededIndices.has(idx)) {
      controller.abort();
      activeFetchControllers.delete(idx);
    }
  });

  // 3. Sort needed photos by priority (Current first, then +1, -1, +2, +3...)
  const prioritizedIndices = Array.from(neededIndices)
    .filter((idx) => idx >= 0 && idx < photos.length)
    .sort((a, b) => {
      const distA = Math.abs(a - currentIndex);
      const distB = Math.abs(b - currentIndex);
      // Give slight preference to forward images over backward ones
      const weightA = a < currentIndex ? distA + 0.1 : distA;
      const weightB = b < currentIndex ? distB + 0.1 : distB;
      return weightA - weightB;
    });

  // 4. Load prioritized photos
  prioritizedIndices.forEach((idx) => {
    const photo = photos[idx];
    if (!photo) return;

    // Skip if already fully loaded or currently downloading
    if (photo.dataset.loaded === "true" || activeFetchControllers.has(idx))
      return;

    const highResSrc = photo.dataset.src;
    const thumbSrc = photo.dataset.thumb;

    // INSTANT: Apply low-res thumbnail immediately so user never sees a black box
    if (!photo.style.backgroundImage && thumbSrc) {
      photo.style.backgroundImage = `url('${thumbSrc}')`;
    }

    // Create AbortController to manage this high-res download
    const controller = new AbortController();
    activeFetchControllers.set(idx, controller);

    fetch(highResSrc, { signal: controller.signal })
      .then((res) => res.blob())
      .then((blob) => {
        const objectUrl = URL.createObjectURL(blob);
        photo.style.backgroundImage = `url('${objectUrl}')`;
        photo.dataset.loaded = "true";
        activeFetchControllers.delete(idx);
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          activeFetchControllers.delete(idx);
        }
      });
  });
}

function setGalleryIndex(i) {
  const photos = document.querySelectorAll(".gallery-photo");
  const thumbs = document.querySelectorAll(".strip-thumb");

  photos.forEach((p, idx) => p.classList.toggle("active", idx === i));
  thumbs.forEach((t, idx) => t.classList.toggle("active", idx === i));

  galleryIndex = i;

  // Trigger smart prioritize/cancel pipeline on navigation
  preloadSmart(i);

  thumbs[i]?.scrollIntoView({
    behavior: "smooth",
    inline: "center",
    block: "nearest",
  });
}

document.getElementById("gallery-prev")?.addEventListener("click", () => {
  const len = window.currentImages ? window.currentImages.length : 0;
  if (len) setGalleryIndex((galleryIndex - 1 + len) % len);
});

document.getElementById("gallery-next")?.addEventListener("click", () => {
  const len = window.currentImages ? window.currentImages.length : 0;
  if (len) setGalleryIndex((galleryIndex + 1) % len);
});

document.getElementById("image-in")?.addEventListener("change", (e) => {
  const file = e.target.files[0];
  document.getElementById("file-label-text").textContent =
    file?.name || "Search by photo";
  if (file) {
    const url = URL.createObjectURL(file);
    document.getElementById("ref-img").src = url;
    document.getElementById("ref-label").textContent = file.name;
    document.getElementById("ref-preview").style.display = "block";
  } else {
    document.getElementById("ref-preview").style.display = "none";
  }
});
// ── RESIZABLE MAP ──
const handle = document.getElementById("resize-handle");
const mapPane = document.getElementById("map-pane");
let isResizing = false;

handle.addEventListener("mousedown", (e) => {
  isResizing = true;
  handle.classList.add("dragging");
  document.body.style.cursor = "col-resize";
  document.body.style.userSelect = "none";
});
document.addEventListener("mousemove", (e) => {
  if (!isResizing) return;
  const split = document.querySelector(".content-split");
  const splitRect = split.getBoundingClientRect();
  const newWidth = splitRect.right - e.clientX;
  const clampedWidth = Math.max(200, Math.min(newWidth, splitRect.width * 0.7));
  mapPane.style.width = clampedWidth + "px";
  if (window.leafletMap) window.leafletMap.invalidateSize();
});
document.addEventListener("mouseup", () => {
  if (!isResizing) return;
  isResizing = false;
  handle.classList.remove("dragging");
  document.body.style.cursor = "";
  document.body.style.userSelect = "";
});
// ── MEMORY ──
let currentMemory = null;

async function saveCollection(memory) {
  const name = `${memory.title} — ${new Date().toLocaleDateString()}`;
  await fetch("/collections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name,
      urls: memory.photos.map((p) => p.url),
    }),
  });
  await renderSavedCollections();
}
async function renderSavedCollections() {
  const resp = await fetch("/collections");
  const { collections } = await resp.json();
  const list = document.getElementById("saved-collections-list");
  if (!list) return;
  list.innerHTML = "";

  if (
    currentMemory &&
    currentMemory.photos &&
    currentMemory.photos.length > 0
  ) {
    const el = document.createElement("div");
    el.style.cssText =
      "background:#f0f7ff;border:1px solid rgba(0,122,255,0.2);border-radius:12px;padding:12px 14px;display:flex;align-items:center;gap:12px;box-shadow:0 1px 3px rgba(0,0,0,0.06);cursor:pointer;transition:background 0.15s;margin-bottom:12px;";
    el.innerHTML = `
<div style="display:flex;gap:3px;flex-shrink:0;">
    ${currentMemory.photos
      .slice(0, 3)
      .map((p) => {
        const url = p.url.replace(/^\//, "");
        return `<div style="width:44px;height:44px;border-radius:8px;background:url('/thumbnail/${url}') center/cover;"></div>`;
      })
      .join("")}
</div>
<div style="flex:1;min-width:0;">
    <div style="font-size:11px;font-weight:600;color:#007aff;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px;">Current</div>
    <div style="font-size:13px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${currentMemory.title}</div>
    <div style="font-size:11px;color:#8e8e93;">${currentMemory.photos.length} photos</div>
</div>
<button id="current-memory-save" style="border:none;background:#007aff;color:white;border-radius:8px;font-size:12px;font-weight:600;padding:6px 10px;cursor:pointer;flex-shrink:0;">Save</button>
`;

    el.addEventListener("click", (e) => {
      if (e.target.closest("button")) return;
      // Force interface state restoration back to the active random generated item view
      document.getElementById("memory-loading").style.display = "none";
      document.getElementById("memory-content").style.display = "block";
      document
        .getElementById("memory-content")
        .scrollIntoView({ behavior: "smooth" });
    });

    el.querySelector("button").addEventListener("click", async (e) => {
      e.stopPropagation();
      await saveCollection(currentMemory);
      e.target.textContent = "✓";
      e.target.style.background = "#34c759";
    });
    list.appendChild(el);
  }

  // ========================================================
  // 2. RENDER THE LIST OF HISTORICAL SAVED COLLECTIONS
  // ========================================================
  if (collections && collections.length > 0) {
    collections.forEach((col) => {
      const el = document.createElement("div");
      el.style.cssText =
        "background:white;border:1px solid rgba(0,0,0,0.06);border-radius:12px;padding:12px 14px;display:flex;align-items:center;gap:12px;box-shadow:0 1px 3px rgba(0,0,0,0.02);cursor:pointer;transition:background 0.15s;margin-bottom:8px;";

      const photoUrlsList = col.urls || [];
      const previewThumbnails = photoUrlsList
        .slice(0, 3)
        .map((urlStr) => {
          const cleanUrl = urlStr.replace(/^\//, "");
          return `<div style="width:44px;height:44px;border-radius:8px;background:url('/thumbnail/${cleanUrl}') center/cover;"></div>`;
        })
        .join("");

      el.innerHTML = `
    <div style="display:flex;gap:3px;flex-shrink:0;">
        ${previewThumbnails || '<div style="width:44px;height:44px;border-radius:8px;background:#e5e5ea;"></div>'}
    </div>
    <div style="flex:1;min-width:0;">
        <div style="font-size:13px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${col.name}</div>
        <div style="font-size:11px;color:#8e8e93;">${photoUrlsList.length || col.count || 0} photos</div>
    </div>
`;

      el.addEventListener("click", () => {
        const collectionPayload = {
          name: col.name,
          count: photoUrlsList.length || col.count || 0,
        };

        openSavedCollection(collectionPayload);

        document
          .getElementById("memory-content")
          .scrollIntoView({ behavior: "smooth" });
      });

      list.appendChild(el);
    });
  }
}
async function openSavedCollection(col) {
  document.getElementById("memory-title").textContent = col.name;
  document.getElementById("memory-subtitle").textContent =
    col.count + " photos";

  document.getElementById("memory-loading").style.display = "flex";
  document.getElementById("memory-content").style.display = "none";
  const resp = await fetch(
    `/collections/${encodeURIComponent(col.name)}/photos`,
  );
  const data = await resp.json();
  const photos = data.photos;

  if (photos.length > 0) {
    const heroUrl = photos[0].url.replace(/^\//, "");
    const hero = document.getElementById("memory-hero");
    hero.style.backgroundImage = `url('/files/${heroUrl}')`;
    hero.onclick = () => window.openLightbox && window.openLightbox(photos, 0);
  }

  const grid = document.getElementById("memory-grid");
  grid.innerHTML = "";
  const selected = new Set();

  function updateCollectionToolbar() {
    const toolbar = document.getElementById("memory-select-toolbar");
    if (toolbar) {
      toolbar.style.display = selected.size > 0 ? "flex" : "none";
      document.getElementById("memory-select-count").textContent =
        `${selected.size} selected`;
    }
  }

  const existingToolbar = document.getElementById("memory-select-toolbar");
  if (existingToolbar) existingToolbar.remove();

  const toolbar = document.createElement("div");
  toolbar.id = "memory-select-toolbar";
  toolbar.style.cssText =
    "display:none;align-items:center;gap:10px;margin-bottom:10px;";
  toolbar.innerHTML = `
<span id="memory-select-count" style="font-size:13px;color:#3c3c43;font-weight:500;"></span>
<button id="memory-deselect-all" style="border:none;background:#f2f2f7;color:#3c3c43;border-radius:8px;padding:6px 12px;font-size:12px;font-weight:600;cursor:pointer;font-family:inherit;">Deselect all</button>
<button id="memory-delete-selected" style="border:none;background:#ff3b30;color:white;border-radius:8px;padding:6px 12px;font-size:12px;font-weight:600;cursor:pointer;font-family:inherit;">✕ Remove selected</button>
`;
  grid.before(toolbar);

  document
    .getElementById("memory-deselect-all")
    .addEventListener("click", () => {
      selected.forEach((url) => {
        const cell = grid.querySelector(`[data-url="${url}"]`);
        if (cell) cell.classList.remove("mem-selected");
      });
      selected.clear();
      updateCollectionToolbar();
    });

  document
    .getElementById("memory-delete-selected")
    .addEventListener("click", async () => {
      if (!confirm(`Remove ${selected.size} photo(s) from this collection?`))
        return;
      for (const url of selected) {
        const cell = grid.querySelector(`[data-url="${url}"]`);
        if (cell) {
          cell.style.opacity = "0";
          cell.style.transform = "scale(0.85)";
          cell.style.transition = "all 0.2s";
        }
        await fetch(`/collections/${encodeURIComponent(col.name)}/remove`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: url }),
        });
      }
      setTimeout(() => {
        selected.forEach((url) =>
          grid.querySelector(`[data-url="${url}"]`)?.remove(),
        );
        selected.clear();
        updateCollectionToolbar();
      }, 220);
    });

  photos.slice(1).forEach((photo, i) => {
    const url = photo.url.replace(/^\//, "");
    const cell = document.createElement("div");
    cell.dataset.url = url;
    cell.style.cssText =
      "aspect-ratio:1;position:relative;border-radius:4px;overflow:hidden;cursor:pointer;transition:opacity 0.2s,transform 0.2s;";
    cell.innerHTML = `
<div style="width:100%;height:100%;background:url('/thumbnail/${url}') center/cover;"></div>
<div class="mem-check" style="display:none;position:absolute;top:5px;left:5px;width:20px;height:20px;border-radius:50%;background:#007aff;border:2px solid white;align-items:center;justify-content:center;color:white;font-size:11px;">✓</div>
<div style="position:absolute;inset:0;border:3px solid transparent;border-radius:4px;transition:border-color 0.15s;" class="mem-border"></div>
`;
    cell.addEventListener("click", (e) => {
      if (selected.size > 0 || e.shiftKey) {
        if (selected.has(url)) {
          selected.delete(url);
          cell.classList.remove("mem-selected");
        } else {
          selected.add(url);
          cell.classList.add("mem-selected");
        }
        updateCollectionToolbar();
      } else {
        if (window.openLightbox) window.openLightbox(photos, i + 1);
      }
    });
    cell.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      selected.add(url);
      cell.classList.add("mem-selected");
      updateCollectionToolbar();
    });
    grid.appendChild(cell);
  });
  // swap save button to "back"
  const saveBtn = document.getElementById("memory-save-btn");
  const origText = "♥ Save Collection";
  saveBtn.textContent = "‹ Back";
  saveBtn.onclick = () => {
    saveBtn.textContent = origText;
    saveBtn.onclick = null;
    loadMemory(); // reload a fresh memory
  };

  document.getElementById("memory-loading").style.display = "none";
  document.getElementById("memory-content").style.display = "block";
}
async function loadMemory() {
  document.getElementById("memory-loading").style.display = "flex";
  document.getElementById("memory-content").style.display = "none";

  try {
    const resp = await fetch("/memory");
    const data = await resp.json();
    currentMemory = data;

    document.getElementById("memory-title").textContent = data.title;
    document.getElementById("memory-subtitle").textContent = data.subtitle;

    if (data.photos.length > 0) {
      const heroUrl = data.photos[0].url.replace(/^\//, "");
      document.getElementById("memory-hero").style.backgroundImage =
        `url('/thumbnail/${heroUrl}?size=800')`;
      const hero = document.getElementById("memory-hero");
      hero.onclick = () => {
        if (window.openLightbox) window.openLightbox(data.photos, 0);
      };
      hero.style.position = "relative";

      const heroDelete = document.createElement("button");
      heroDelete.textContent = "✕";
      heroDelete.style.cssText =
        "position:absolute;top:10px;right:10px;border:none;background:rgba(255,59,48,0.8);color:white;border-radius:50%;width:30px;height:30px;font-size:14px;cursor:pointer;z-index:10;";
      heroDelete.onclick = async (e) => {
        e.stopPropagation();
        if (!confirm("Permanently delete this photo from disk?")) return;
        const url = data.photos[0].url.replace(/^\//, "");
        const fd = new FormData();
        fd.append("image_paths", url);
        await fetch("/delete_photo", { method: "POST", body: fd });
        data.photos.splice(0, 1);
        loadMemory();
      };
      hero.appendChild(heroDelete);

      const grid = document.getElementById("memory-grid");
      grid.innerHTML = "";
      // multiselect state
      const selected = new Set();

      // toolbar
      const toolbar = document.createElement("div");
      toolbar.id = "memory-select-toolbar";
      toolbar.style.cssText =
        "display:none;align-items:center;gap:10px;margin-bottom:10px;";
      toolbar.innerHTML = `
<span id="memory-select-count" style="font-size:13px;color:#3c3c43;font-weight:500;"></span>
<button id="memory-deselect-all" style="border:none;background:#f2f2f7;color:#3c3c43;border-radius:8px;padding:6px 12px;font-size:12px;font-weight:600;cursor:pointer;font-family:inherit;">Deselect all</button>
<button id="memory-delete-selected" style="border:none;background:#ff3b30;color:white;border-radius:8px;padding:6px 12px;font-size:12px;font-weight:600;cursor:pointer;font-family:inherit;">✕ Remove selected</button>
`;
      grid.before(toolbar);

      function updateToolbar() {
        toolbar.style.display = selected.size > 0 ? "flex" : "none";
        document.getElementById("memory-select-count").textContent =
          `${selected.size} selected`;
      }

      document
        .getElementById("memory-deselect-all")
        .addEventListener("click", () => {
          selected.forEach((url) => {
            const cell = grid.querySelector(`[data-url="${url}"]`);
            if (cell) cell.classList.remove("mem-selected");
          });
          selected.clear();
          updateToolbar();
        });

      document
        .getElementById("memory-delete-selected")
        .addEventListener("click", async () => {
          if (
            !confirm(`Permanently delete ${selected.size} photo(s) from disk?`)
          )
            return;
          for (const url of selected) {
            const cell = grid.querySelector(`[data-url="${url}"]`);
            if (cell) {
              cell.style.opacity = "0";
              cell.style.transform = "scale(0.85)";
              cell.style.transition = "all 0.2s";
            }
            const fd = new FormData();
            fd.append("image_paths", url);
            await fetch("/delete_photo", { method: "POST", body: fd });
            const idx = data.photos.findIndex(
              (p) => p.url.replace(/^\//, "") === url,
            );
            if (idx >= 0) data.photos.splice(idx, 1);
          }
          setTimeout(() => {
            selected.forEach((url) =>
              grid.querySelector(`[data-url="${url}"]`)?.remove(),
            );
            selected.clear();
            updateToolbar();
          }, 220);
        });
      if (!document.getElementById("mem-selected-style")) {
        const style = document.createElement("style");
        style.id = "mem-selected-style";
        style.textContent = `.mem-selected .mem-check { display:flex !important; } .mem-selected .mem-border { border-color: #007aff !important; } .mem-selected { opacity: 0.85; }`;
        document.head.appendChild(style);
      }
      data.photos.slice(1).forEach((photo, i) => {
        const url = photo.url.replace(/^\//, "");
        const cell = document.createElement("div");
        cell.dataset.url = url;
        cell.style.cssText =
          "aspect-ratio:1;position:relative;border-radius:4px;overflow:hidden;cursor:pointer;transition:opacity 0.2s,transform 0.2s;";
        cell.innerHTML = `
    <div style="width:100%;height:100%;background:url('/thumbnail/${url}') center/cover;"></div>
    <div class="mem-check" style="display:none;position:absolute;top:5px;left:5px;width:20px;height:20px;border-radius:50%;background:#007aff;border:2px solid white;align-items:center;justify-content:center;color:white;font-size:11px;">✓</div>
    <div style="position:absolute;inset:0;border:3px solid transparent;border-radius:4px;transition:border-color 0.15s;" class="mem-border"></div>
`;

        cell.addEventListener("click", (e) => {
          if (selected.size > 0 || e.shiftKey) {
            // in select mode — toggle
            if (selected.has(url)) {
              selected.delete(url);
              cell.classList.remove("mem-selected");
            } else {
              selected.add(url);
              cell.classList.add("mem-selected");
            }
            updateToolbar();
          } else {
            // normal mode — open lightbox
            if (window.openLightbox) window.openLightbox(data.photos, i + 1);
          }
        });

        cell.addEventListener("contextmenu", (e) => {
          e.preventDefault();
          selected.add(url);
          cell.classList.add("mem-selected");
          updateToolbar();
        });

        grid.appendChild(cell);
      });

      await renderSavedCollections();

      document.getElementById("memory-loading").style.display = "none";
      document.getElementById("memory-content").style.display = "block";
    }
  } catch (err) {
    document.getElementById("memory-loading").innerHTML =
      '<div style="font-size:14px;color:#ff3b30">Failed to load memory</div>';
    console.error("loadMemory error:", err);
  }
}

document
  .getElementById("memory-save-btn")
  .addEventListener("click", async () => {
    if (currentMemory) {
      await saveCollection(currentMemory);
      document.getElementById("memory-save-btn").textContent = "✓ Saved!";
      setTimeout(() => {
        document.getElementById("memory-save-btn").textContent =
          "♥ Save Collection";
      }, 2000);
    }
  });

document.getElementById("memory-delete-btn").addEventListener("click", () => {
  loadMemory(); // dismiss = load a new one
});

document.getElementById("memory-refresh-btn").addEventListener("click", () => {
  loadMemory();
});

// ── ALBUMS ──
async function loadAlbums() {
  const resp = await fetch("/albums");
  const data = await resp.json();
  renderAlbumList(data.albums);
}

function renderAlbumList(albums) {
  const list = document.getElementById("album-list");
  list.innerHTML = "";
  if (!albums.length) {
    list.innerHTML =
      '<span style="font-size:12px;color:#8e8e93">No albums yet</span>';
    return;
  }
  albums.forEach((album) => {
    const el = document.createElement("div");
    el.className = "album-row";
    el.draggable = false; // row itself not draggable, it's a drop target
    el.dataset.name = album.name;
    el.style.cssText = `
display:flex;align-items:center;gap:8px;
padding:7px 10px;border-radius:10px;
background:white;cursor:pointer;
box-shadow:0 1px 3px rgba(0,0,0,0.06);
transition:background 0.15s;
position:relative;
`;
    el.innerHTML = `
<span style="font-size:16px;">🗂</span>
<div style="flex:1;min-width:0;">
    <div class="album-name" style="font-size:13px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${album.name}</div>
    <div style="font-size:10px;color:#8e8e93;">${album.count} photos</div>
</div>
<button class="album-del" style="
    border:none;background:none;color:#c7c7cc;
    font-size:14px;cursor:pointer;padding:2px 4px;
    opacity:0;transition:opacity 0.15s;
">✕</button>
`;

    // single click → open overlay
    el.addEventListener("click", (e) => {
      if (e.target.classList.contains("album-del")) return;
      openAlbumOverlay(album.name);
    });

    // double click → rename
    el.addEventListener("dblclick", (e) => {
      if (e.target.classList.contains("album-del")) return;
      const nameEl = el.querySelector(".album-name");
      const old = nameEl.textContent;
      nameEl.contentEditable = true;
      nameEl.focus();
      const range = document.createRange();
      range.selectNodeContents(nameEl);
      window.getSelection().removeAllRanges();
      window.getSelection().addRange(range);

      nameEl.onblur = async () => {
        nameEl.contentEditable = false;
        const newName = nameEl.textContent.trim();
        if (newName && newName !== old) {
          await fetch(`/albums/${encodeURIComponent(old)}/rename`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ new_name: newName }),
          });
        }
        loadAlbums();
      };
      nameEl.onkeydown = (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          nameEl.blur();
        }
      };
    });

    // delete button
    el.querySelector(".album-del").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete album "${album.name}"?`)) return;
      await fetch(`/albums/${encodeURIComponent(album.name)}`, {
        method: "DELETE",
      });
      loadAlbums();
    });

    // show/hide delete on hover
    el.addEventListener(
      "mouseenter",
      () => (el.querySelector(".album-del").style.opacity = "1"),
    );
    el.addEventListener(
      "mouseleave",
      () => (el.querySelector(".album-del").style.opacity = "0"),
    );

    // drop target for drag from grid
    el.addEventListener("dragover", (e) => {
      e.preventDefault();
      el.style.background = "#e8f0fe";
    });
    el.addEventListener("dragleave", () => (el.style.background = "white"));
    el.addEventListener("drop", async (e) => {
      e.preventDefault();
      el.style.background = "white";
      const raw = e.dataTransfer.getData("photo-paths");
      if (!raw) return;
      const paths = JSON.parse(raw);
      await Promise.all(
        paths.map((path) =>
          fetch(`/albums/${encodeURIComponent(album.name)}/add`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path }),
          }),
        ),
      );
      // flash feedback

      el.style.background = "#d4edda";
      el.querySelector(".album-name").textContent += ` (+${paths.length})`;
      setTimeout(() => {
        el.style.background = "white";
        loadAlbums();
      }, 600);
    });

    list.appendChild(el);
  });
}

// new album button
document.getElementById("new-album-btn").addEventListener("click", async () => {
  const name = prompt("Album name:");
  if (!name) return;

  await fetch("/albums", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  loadAlbums();
});

// ── ALBUM OVERLAY ──
async function openAlbumOverlay(name) {
  document.getElementById("album-overlay-title").textContent = name;
  document.getElementById("album-overlay-count").textContent = "";
  document.getElementById("album-overlay-grid").innerHTML = "";
  document.getElementById("album-overlay").style.display = "flex";

  const resp = await fetch(`/albums/${encodeURIComponent(name)}/photos`);
  const data = await resp.json();
  document.getElementById("album-overlay-count").textContent =
    `${data.photos.length} photos`;

  const grid = document.getElementById("album-overlay-grid");
  data.photos.forEach((photo, i) => {
    const url = photo.url.replace(/^\//, "");
    const cell = document.createElement("div");
    cell.style.cssText =
      "aspect-ratio:1;position:relative;border-radius:4px;overflow:hidden;";
    cell.innerHTML = `
<div style="width:100%;height:100%;background:url('/thumbnail/${url}') center/cover;cursor:pointer;"></div>
<div class="grid-overlay" style="position:absolute;inset:0;background:linear-gradient(160deg,rgba(0,0,0,0.35) 0%,transparent 45%);opacity:0;transition:opacity 0.18s;padding:7px;display:flex;justify-content:space-between;align-items:flex-start;">
    <button class="thumb-btn thumb-fav" title="Favorite">♥</button>
    <button class="thumb-btn thumb-delete" title="Delete">✕</button>
</div>
`;
    cell
      .querySelector("div")
      .addEventListener("click", () => window.openLightbox(data.photos, i));
    cell.addEventListener(
      "mouseenter",
      () => (cell.querySelector(".grid-overlay").style.opacity = "1"),
    );
    cell.addEventListener(
      "mouseleave",
      () => (cell.querySelector(".grid-overlay").style.opacity = "0"),
    );

    cell.querySelector(".thumb-fav").addEventListener("click", (e) => {
      e.stopPropagation();
      const isFav = window.toggleFavorite(url);
      e.target.classList.toggle("fav-active", isFav);
    });
    cell
      .querySelector(".thumb-fav")
      .classList.toggle("fav-active", window.getFavorites().includes(url));

    cell.querySelector(".thumb-delete").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm("Delete this photo?")) return;
      const fd = new FormData();
      fd.append("image_paths", url);
      await fetch("/delete_photo", { method: "POST", body: fd });
      cell.style.opacity = "0";
      setTimeout(() => cell.remove(), 300);
    });

    grid.appendChild(cell);
  });
}

document.getElementById("album-back").addEventListener("click", () => {
  document.getElementById("album-overlay").style.display = "none";
});
// ── REVIEW ──
function getSeenPhotos() {
  try {
    return new Set(JSON.parse(localStorage.getItem("review_seen") || "[]"));
  } catch {
    return new Set();
  }
}
function addSeenPhotos(urls) {
  const seen = getSeenPhotos();
  urls.forEach((u) => seen.add(u));
  localStorage.setItem("review_seen", JSON.stringify([...seen]));
}
function resetSeenPhotos() {
  localStorage.removeItem("review_seen");
}

let reviewPhotos = [];

async function loadReviewBatch() {
  const seen = getSeenPhotos();
  document.getElementById("review-exhausted").style.display = "none";
  document.getElementById("review-grid").style.display = "grid";

  const resp = await fetch(
    `/review/batch?seen=${encodeURIComponent([...seen].join(","))}`,
  );
  const data = await resp.json();

  if (data.exhausted || data.photos.length === 0) {
    document.getElementById("review-exhausted").style.display = "flex";
    document.getElementById("review-grid").style.display = "none";
    return;
  }

  reviewPhotos = data.photos;
  addSeenPhotos(data.photos.map((p) => p.url.replace(/^\//, "")));

  const total = getSeenPhotos().size;
  document.getElementById("review-progress").textContent = `${total} reviewed`;

  const grid = document.getElementById("review-grid");
  grid.innerHTML = "";
  loadFavoritesFromDisk();
  data.photos.forEach((photo, i) => {
    const url = photo.url.replace(/^\//, "");
    const cell = document.createElement("div");
    const favs = _favSet ? [..._favSet] : [];
    const isFav = favs.includes(url);
    cell.style.cssText =
      "aspect-ratio:1;position:relative;border-radius:4px;overflow:hidden;opacity:0;transform:scale(0.94);transition:opacity 0.3s ease,transform 0.3s ease;";
    cell.innerHTML = `
<div style="width:100%;height:100%;background:url('/thumbnail/${url}') center/cover;cursor:pointer;"></div>
<div style="position:absolute;inset:0;background:linear-gradient(160deg,rgba(0,0,0,0.35) 0%,transparent 45%);opacity:0;transition:opacity 0.18s;padding:7px;display:flex;justify-content:flex-end;align-items:flex-start;">
    <button class="thumb-btn thumb-fav ${isFav ? "fav-active" : ""}" title="Favorite">♥</button>
    <button class="thumb-btn thumb-delete" title="Delete">✕</button>
</div>
`;

    cell.querySelector("div").addEventListener("click", (e) => {
      if (e.target.closest(".thumb-btn")) return;
      window.openLightbox(reviewPhotos, i, false);
    });

    const overlay = cell.querySelector("div + div");
    cell.addEventListener("mouseenter", () => (overlay.style.opacity = "1"));
    cell.addEventListener("mouseleave", () => (overlay.style.opacity = "0"));
    cell.querySelector(".thumb-fav").addEventListener("click", (e) => {
      e.stopPropagation();
      const nowFav = toggleFavorite(url);
      e.currentTarget.classList.toggle("fav-active", nowFav);
    });
    cell.querySelector(".thumb-delete").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm("Delete this photo?")) return;
      const fd = new FormData();
      fd.append("image_paths", url);
      await fetch("/delete_photo", { method: "POST", body: fd });
      cell.style.opacity = "0";
      cell.style.transform = "scale(0.85)";
      setTimeout(() => cell.remove(), 300);
    });

    grid.appendChild(cell);

    // staggered appear
    requestAnimationFrame(() => {
      setTimeout(
        () => {
          cell.style.opacity = "1";
          cell.style.transform = "scale(1)";
        },
        Math.min(i * 30, 500),
      );
    });
  });
}

document
  .getElementById("review-next-btn")
  .addEventListener("click", loadReviewBatch);
document.getElementById("review-reset-btn").addEventListener("click", () => {
  resetSeenPhotos();
  loadReviewBatch();
});

// hook into setView
loadAlbums();
window.renderFavStrip = renderFavStrip;
window.getFavorites = getFavorites; // move this up too
window.toggleFavorite = toggleFavorite;
window.addEventListener("storage", (e) => {
  if (e.key === "mem_favorites") window.renderFavStrip();
});
document.addEventListener("DOMContentLoaded", () => {
  const interval = setInterval(() => {
    if (window.renderFavStrip) {
      window.renderFavStrip();
      clearInterval(interval);
    }
  }, 100);
});

function updateGridToolbar() {
  const toolbar = document.getElementById("grid-select-toolbar");
  toolbar.style.display = gridSelected.size > 0 ? "flex" : "none";
  document.getElementById("grid-select-count").textContent =
    `${gridSelected.size} selected`;
}

function clearGridSelection() {
  gridSelected.forEach((url) => {
    const cell = document.querySelector(`.grid-cell[data-url="${url}"]`);
    if (cell) cell.classList.remove("grid-selected");
  });
  gridSelected.clear();
  updateGridToolbar();
}

document
  .getElementById("grid-deselect-all")
  .addEventListener("click", clearGridSelection);

document.getElementById("grid-fav-selected").addEventListener("click", () => {
  gridSelected.forEach((url) => window.toggleFavorite(url));
  document.getElementById("grid-fav-selected").textContent = "✓ Done";
  setTimeout(() => {
    document.getElementById("grid-fav-selected").textContent = "♥ Favorite";
    clearGridSelection();
  }, 800);
});

document.getElementById("clear-btn").addEventListener("click", () => {
  console.log("clear clicked");
  document.getElementById("start-date").value = "";
  document.getElementById("end-date").value = "";
  document.getElementById("country-filter").value = "";
  document.getElementById("city-filter").value = "";
  document.getElementById("h3-filter").value = "";
  document.getElementById("limit").value = "";
  const fileInput = document.getElementById("image-in");
  fileInput.value = "";
  try {
    fileInput.value = null;
  } catch (e) {}
  document.getElementById("file-label-text").textContent = "Search by photo";
  document.getElementById("ref-preview").style.display = "none";
  if (window._clearFaceSelection) window._clearFaceSelection();
});
async function deleteSelectedPhotos() {
  if (gridSelected.size === 0) return;

  const count = gridSelected.size;
  if (
    !confirm(
      `Are you sure you want to permanently delete these ${count} photos from your disk?`,
    )
  )
    return;
  window.loadingBar = loadingBar;
  window.loadingText = loadingText;
  document.getElementById("loading-container").style.display = "block";
  if (window.loadingText)
    window.loadingText.textContent = `Deleting ${count} photos...`;

  for (const url of [...gridSelected]) {
    try {
      const fd = new FormData();
      fd.append("image_paths", url);

      await fetch("/delete_photo", {
        method: "POST",
        body: fd,
      });

      if (_favSet && _favSet.has(url)) {
        _favSet.delete(url);
      }

      const cell = document.querySelector(`[data-url="${CSS.escape(url)}"]`);
      if (cell) {
        cell.style.opacity = "0";
        cell.style.transform = "scale(0.85)";
        setTimeout(() => cell.remove(), 300);
      }

      window.currentImages = window.currentImages.filter(
        (img) => img.url.replace(/^\//, "") !== url,
      );
      gridSelected.delete(url);
    } catch (err) {
      console.error(`Failed to delete photo: ${url}`, err);
    }
  }

  // Refresh UI trackers
  renderFavStrip();
  updateGridToolbar();

  document.getElementById("loading-container").style.display = "none";
}

// Expose it to your global HTML window context
window.deleteSelectedPhotos = deleteSelectedPhotos;
document
  .getElementById("grid-delete-selected")
  .addEventListener("click", async () => {
    if (!confirm(`Permanently delete ${gridSelected.size} photo(s)?`)) return;
    for (const url of gridSelected) {
      const fd = new FormData();
      fd.append("image_paths", url);
      await fetch("/delete_photo", { method: "POST", body: fd });
      const cell = document.querySelector(`.grid-cell[data-url="${url}"]`);
      if (cell) {
        cell.style.opacity = "0";
        cell.style.transform = "scale(0.85)";
        cell.style.transition = "all 0.2s";
      }
    }
    setTimeout(() => {
      gridSelected.forEach((url) =>
        document.querySelector(`.grid-cell[data-url="${url}"]`)?.remove(),
      );
      clearGridSelection();
    }, 220);
  });
