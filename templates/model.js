import { InferenceSession, Tensor } from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.min.mjs';
          import { FaceDetector, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs';

          const loadingBar = document.getElementById('loading-bar');
          const loadingText = document.getElementById('loading-text');
          const btn = document.querySelector('button[name="forwardBtn"]');
          btn.disabled = true;
          let dinoSession = null;
          function renderFavStrip() {
              const strip = document.getElementById('fav-strip');
              strip.innerHTML = '';
              const favs = getFavorites();
              if (favs.length === 0) {
                  strip.innerHTML = '<span style="font-size:12px;color:#8e8e93">No favorites yet</span>';
                  return;
              }
              favs.forEach(url => {
                  const el = document.createElement('div');
                  el.className = 'fav-thumb';
                  el.style.backgroundImage = `url('/thumbnail/${url}')`;
                  el.title = url.split('/').pop();
                  el.addEventListener('click', () => {
                      const allFavs = getFavorites().map(u => ({ url: u, score: 0, lat: null, lon: null, location: null }));
                      const i = allFavs.findIndex(img => img.url === url);
                      openLightbox(allFavs, i >= 0 ? i : 0);
                  });
                  strip.appendChild(el);
              });
          }
          window.renderFavStrip = renderFavStrip;
          async function fetchWithProgress(url, onProgress) {
              const response = await fetch(url)
              const contentLength = response.headers.get('Content-Length');
              const total = contentLength ? parseInt(contentLength) : null;
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
          loadingText.textContent = 'Loading DINO...';
          loadingBar.style.width = '0%';

          const dinoBuffer = await fetchWithProgress(
          'https://huggingface.co/georgerieh/onnx-dino-vitb-16-and-facenet/resolve/main/dinov2_vitb14.onnx',
          (pct) => {
              const display = Math.round(pct * 20); // 0–20% allocation
              loadingBar.style.width = display + '%';
              loadingText.textContent = `Loading DINO Graph... ${display}%`;
          }
          );

          loadingText.textContent = 'Loading DINO Weights...';

          const dinoDataBuffer = await fetchWithProgress(
              'https://huggingface.co/georgerieh/onnx-dino-vitb-16-and-facenet/resolve/main/dinov2_vitb14.onnx.data',
              (pct) => {
                  const display = 20 + Math.round(pct * 25); // 20–45% allocation
                  loadingBar.style.width = display + '%';
                  loadingText.textContent = `Loading DINO Weights... ${display}%`;
              }
          );

          dinoSession = await InferenceSession.create(dinoBuffer, {
              executionProviders: ['webgpu', 'wasm'],
              externalData: [
                  {
                      data: new Uint8Array(dinoDataBuffer),
                      path: 'dinov2_vitb14.onnx.data'
                  }
              ]
          });

          loadingBar.style.width = '50%';
          loadingText.textContent = 'Loading FaceNet...';

          const facenetBuffer = await fetchWithProgress('https://huggingface.co/georgerieh/onnx-dino-vitb-16-and-facenet/resolve/main/facenet_inline.onnx', (pct) => {
              const display = 50 + Math.round(pct * 40); // FaceNet occupies 50–90%
              loadingBar.style.width = display + '%';
              loadingText.textContent = `Loading FaceNet... ${display}%`;
          });

          const faceNetSession = await InferenceSession.create(facenetBuffer, {
              executionProviders: ['webgpu', 'wasm'],
          });

          loadingBar.style.width = '95%';
          loadingText.textContent = 'Loading face detector...';
          const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm');
          const faceDetector = await FaceDetector.createFromOptions(vision, {
              baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite' },
              runningMode: 'IMAGE',
              minDetectionConfidence: 0.9,
          });

          loadingBar.style.width = '100%';
          loadingText.textContent = '✓ Ready — upload a photo to search';
          btn.disabled = false;
          setTimeout(() => {
              loadingBar.style.width = '0%';
              loadingText.textContent = '';
              document.getElementById('loading-container').style.display = 'none';
          }, 2000); // longer so user sees it's ready

          function preprocessDino(imgElement) {
              const canvas = document.createElement('canvas');
              canvas.width = 224; canvas.height = 224;
              const ctx = canvas.getContext('2d');
              ctx.drawImage(imgElement, 0, 0, 224, 224);
              const { data } = ctx.getImageData(0, 0, 224, 224);
              const tensor = new Float32Array(3 * 224 * 224);
              for (let i = 0; i < 224 * 224; i++) {
                  tensor[i]           = (data[i*4]   /255.0-0.5)/0.5;
                  tensor[i+224*224]   = (data[i*4+1] /255.0-0.5)/0.5;
                  tensor[i+2*224*224] = (data[i*4+2] /255.0-0.5)/0.5;
              }
              return new Tensor('float32', tensor, [1,3,224,224]);
          }

          function preprocessFace(imgElement, box) {
              const canvas = document.createElement('canvas');
              canvas.width = 160; canvas.height = 160;
              const ctx = canvas.getContext('2d');
              ctx.drawImage(imgElement, box.originX, box.originY, box.width, box.height, 0, 0, 160, 160);
              const { data } = ctx.getImageData(0, 0, 160, 160);
              const tensor = new Float32Array(3 * 160 * 160);
              for (let i = 0; i < 160 * 160; i++) {
                  tensor[i]           = (data[i*4]   /255.0-0.5)/0.5;
                  tensor[i+160*160]   = (data[i*4+1] /255.0-0.5)/0.5;
                  tensor[i+2*160*160] = (data[i*4+2] /255.0-0.5)/0.5;
              }
              return new Tensor('float32', tensor, [1,3,160,160]);
          }

          function normalize(arr) {
              const norm = Math.sqrt(arr.reduce((s,v) => s+v*v, 0));
              return arr.map(v => v/norm);
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
              leafletMap = L.map('map').setView([48.505, 2.33], 3);
              L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 18, attribution: '© OpenStreetMap' }).addTo(leafletMap);
              leafletMap.on('zoomend moveend', updateHexLayer);
              updateHexLayer();
          }
          let hexLayer = null;

              function updateHexLayer() {
                  if (!leafletMap) return;
                  const zoom = leafletMap.getZoom();
                  // map zoom to h3 resolution
                  const res = zoom < 4 ? 3 : zoom < 6 ? 4 : zoom < 8 ? 5 : zoom < 10 ? 6 : 7;

                  fetch(`/hex_coverage?resolution=${res}`)
                      .then(r => r.json())
                      .then(data => {
                          if (hexLayer) leafletMap.removeLayer(hexLayer);
                          const layers = [];
                          data.cells.forEach(cell => {
                              const boundary = h3.cellToBoundary(cell.h3).map(([lat, lng]) => [lat, lng]);
                              const poly = L.polygon(boundary, {
                                  color: '#007aff',
                                  fillColor: '#007aff',
                                  fillOpacity: 0.08,
                                  weight: 1,
                                  opacity: 0.4,
                              });
                              poly.bindTooltip(`${cell.count} photos`, { sticky: true });
                                      poly.on('click', () => {
                     const h3id = cell.h3;
                      try {
                          navigator.clipboard.writeText(h3id);
                      } catch {
                          // fallback for non-https
                          const ta = document.createElement('textarea');
                          ta.value = h3id;
                          document.body.appendChild(ta);
                          ta.select();
                          document.execCommand('copy');
                          document.body.removeChild(ta);
                      }
                      // also put in h3 filter input
                      document.getElementById('h3-filter').value = h3id;
                      poly.setStyle({ fillOpacity: 0.25 });
                      poly.bindTooltip(`Copied: ${h3id}`, {sticky: true}).openTooltip();
                      setTimeout(() => poly.setStyle({ fillOpacity: 0.08 }), 600);
                  });
                              layers.push(poly);
                          });
                          hexLayer = L.layerGroup(layers).addTo(leafletMap);
                      });
              }
          initMap();
          document.querySelector('form[name="input"]').addEventListener('submit', async (e) => {
          e.preventDefault();
          const fileInput = document.getElementById('image-in');
          const startDate = document.getElementById('start-date').value;
          const endDate = document.getElementById('end-date').value;
          const limit = document.getElementById('limit').value || 50;
          const country = document.getElementById('country-filter').value;
          const city = document.getElementById('city-filter').value;
          const h3cell = document.getElementById('h3-filter').value;

          const hasImage = fileInput.files.length > 0;
          const hasFilters = startDate || endDate || country || city || h3cell;

          if (!hasImage && !hasFilters && !selectedFaceEmbedding) return;

          document.getElementById('loading-container').style.display = 'block';
          window.currentImages = [];
          document.getElementById('photo-grid').innerHTML = '';
          clearGridSelection();

          let embedding = null, facenetEmbedding = null;

          if (hasImage) {
              loadingBar.style.width = '0%';
              btn.disabled = true;
              const img = await loadImage(fileInput.files[0]);
              loadingText.textContent = 'Computing visual embedding...';
              loadingBar.style.width = '30%';
              const dinoInputName = dinoSession.inputNames[0];
              const dinoFeeds = {};
              dinoFeeds[dinoInputName] = preprocessDino(img);

              const dinoResults = await dinoSession.run(dinoFeeds);
              const dinoOutputName = dinoSession.outputNames[0];
              embedding = normalize(Array.from(dinoResults[dinoOutputName].data).slice(0, 768));
              loadingBar.style.width = '60%';
              if (document.getElementById('detect-faces').checked) {
              console.log("Preparing face detection framework...");
              loadingText.textContent = 'Detecting faces...';

              let detection = { detections: [] };

              try {
                  if (!faceDetector) {
                      throw new Error("MediaPipe FaceDetector instance is not initialized.");
                  }

                  console.log("Passing image to MediaPipe WASM runtime...", img);
                  detection = faceDetector.detect(img);
                  console.log("MediaPipe completed successfully. Found faces:", detection.detections.length);

              } catch (e) {
                  console.error('Face detection step crashed or timed out:', e);
                  detection = { detections: [] };
              }

              if (detection.detections && detection.detections.length > 0) {
                  loadingText.textContent = 'Recognizing face...';
                  loadingBar.style.width = '80%';

                  const best = detection.detections.reduce((a, b) =>
                      a.categories[0].score > b.categories[0].score ? a : b
                  );

                  const facenetInputName = faceNetSession.inputNames[0];
                  const facenetFeeds = {};
                  facenetFeeds[facenetInputName] = preprocessFace(img, best.boundingBox);

                  const faceResults = await faceNetSession.run(facenetFeeds);
                  const facenetOutputName = faceNetSession.outputNames[0];
                  facenetEmbedding = normalize(Array.from(faceResults[facenetOutputName].data));

                  const cropCanvas = document.createElement('canvas');
                  cropCanvas.width = 160; cropCanvas.height = 160;
                  const cropCtx = cropCanvas.getContext('2d');
                  const box = best.boundingBox;
                  cropCtx.drawImage(img, box.originX, box.originY, box.width, box.height, 0, 0, 160, 160);

                  const isKnown = savedFaces.some(f => {
                      const dot = f.embedding.reduce((s,v,i) => s + v * facenetEmbedding[i], 0);
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
              loadingBar.style.width = '50%';
              loadingText.textContent = 'Searching...';
          }

          const response = await fetch('/search_stream', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({
                  embedding,
                  facenet_embedding: facenetEmbedding || selectedFaceEmbedding,
                  start_date: startDate, end_date: endDate, limit,
                  country, city, h3cell
              })
          });
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';
          while (true) {
              const {done, value} = await reader.read();
              if (done) break;
              buffer += decoder.decode(value, {stream: true});
              const lines = buffer.split('\n');
              buffer = lines.pop();
              for (const line of lines) {
                  if (!line) continue;
                  const chunk = JSON.parse(line);
                  window.renderChunk(chunk.images);
              }
          }
          loadingBar.style.width = '100%';
          loadingText.textContent = `${window.currentImages.length} results`;
          setTimeout(() => { document.getElementById('loading-container').style.display = 'none'; }, 600);
          setView('grid');
      });
          document.getElementById('btn-open-face-upload').addEventListener('click', () => {
          document.getElementById('face-upload-modal').style.display = 'flex';
      });

      document.getElementById('face-upload-close').addEventListener('click', () => {
          document.getElementById('face-upload-modal').style.display = 'none';
      });

      const dropZone = document.getElementById('face-drop-zone');
      const faceFileInput = document.getElementById('face-file-input');

      dropZone.addEventListener('click', () => faceFileInput.click());

      dropZone.addEventListener('dragover', (e) => {
          e.preventDefault();
          dropZone.style.borderColor = '#007aff';
          dropZone.style.background = '#f0f7ff';
      });

      dropZone.addEventListener('dragleave', () => {
          dropZone.style.borderColor = '#c7c7cc';
          dropZone.style.background = '#fafafa';
      });

      dropZone.addEventListener('drop', (e) => {
          e.preventDefault();
          dropZone.style.borderColor = '#c7c7cc';
          dropZone.style.background = '#fafafa';
          if (e.dataTransfer.files.length) {
              processFaceImage(e.dataTransfer.files[0]);
          }
      });

      faceFileInput.addEventListener('change', (e) => {
          if (e.target.files.length) {
              processFaceImage(e.target.files[0]);
          }
      });

      async function processFaceImage(file) {
          const loadingText = document.getElementById('face-upload-loading');
          loadingText.style.display = 'block';
          dropZone.style.display = 'none';

          try {
              const img = await loadImage(file);

              let detection = { detections: [] };
              if (faceDetector) {
                  detection = faceDetector.detect(img);
              }

              if (detection.detections && detection.detections.length > 0) {
                  // Find the most confident face
                  const best = detection.detections.reduce((a, b) =>
                      a.categories[0].score > b.categories[0].score ? a : b
                  );

                  // Run FaceNet
                  const facenetInputName = faceNetSession.inputNames[0];
                  const facenetFeeds = {};
                  facenetFeeds[facenetInputName] = preprocessFace(img, best.boundingBox);

                  const faceResults = await faceNetSession.run(facenetFeeds);
                  const facenetOutputName = faceNetSession.outputNames[0];
                  const facenetEmbedding = normalize(Array.from(faceResults[facenetOutputName].data));

                  // Crop it for the thumbnail
                  const cropCanvas = document.createElement('canvas');
                  cropCanvas.width = 160; cropCanvas.height = 160;
                  const cropCtx = cropCanvas.getContext('2d');
                  const box = best.boundingBox;
                  cropCtx.drawImage(img, box.originX, box.originY, box.width, box.height, 0, 0, 160, 160);

                  // Hide upload modal and trigger your existing save face modal
                  document.getElementById('face-upload-modal').style.display = 'none';
                  showFaceModal(facenetEmbedding, cropCanvas, 60000); // Allow 60s to type a name
              } else {
                  alert("No face detected in this image. Try a clearer photo.");
              }
          } catch (err) {
              console.error("Face processing failed:", err);
              alert("An error occurred while processing the face.");
          } finally {
              // Reset UI
              loadingText.style.display = 'none';
              dropZone.style.display = 'flex';
              faceFileInput.value = '';
          }
      }
          let _favSet = null;

          async function loadFavoritesFromDisk() {
              const resp = await fetch('/favorites');
              const { photos } = await resp.json();
              _favSet = new Set(photos.map(p => cleanPhotoUrl(p.url)));
              const stored = JSON.parse(localStorage.getItem('mem_favorites') || '[]');
              const unsaved = stored.map(u => cleanPhotoUrl(u)).filter(u => !_favSet.has(u));

              if (unsaved.length > 0) {
                  const save = confirm(`You have ${unsaved.length} favorite(s) saved only in browser storage. Save them to disk permanently?`);
                  if (save) {
                      for (const url of unsaved) {
                          await fetch('/favorites/add', {
                              method: 'POST',
                              headers: {'Content-Type':'application/json'},
                              body: JSON.stringify({path: url})
                          });
                          _favSet.add(url);
                      }
                      localStorage.removeItem('mem_favorites');
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
                  await fetch('/favorites/remove', {
                      method: 'DELETE',
                      headers: {'Content-Type':'application/json'},
                      body: JSON.stringify({path: cleanUrl})
                  });
              } else {
                  _favSet.add(cleanUrl);
                  await fetch('/favorites/add', {
                      method: 'POST',
                      headers: {'Content-Type':'application/json'},
                      body: JSON.stringify({path: cleanUrl})
                  });
              }
              renderFavStrip();
              return _favSet.has(cleanUrl);
          }

          // ── LIGHTBOX ──
          let lightboxImages = [], lightboxIndex = 0;

          function openLightbox(images, index) {
              lightboxImages = images;
              lightboxIndex = index;
              updateLightbox();
              document.getElementById('lightbox').classList.add('open');
              document.body.style.overflow = 'hidden';
          }
          function closeLightbox() {
              document.getElementById('lightbox').classList.remove('open');
              document.body.style.overflow = '';
          }
          const BASE_PATH = 'media/georgerieh/T7/photos_from_icloud/';
          const BASE = BASE_PATH
          function cleanPhotoUrl(url) {
          if (!url) return '';

          let clean = decodeURIComponent(url).replace(/^\//, '');

          const prefixesToRemove = [
              'files/',
              'thumbnail/',
              'static/',
              'media/georgerieh/T7/photos_from_icloud/',
              'media/georgerieh/T7/'
          ];

          let changed = true;
          while (changed) {
              changed = false;
              for (const prefix of prefixesToRemove) {
                  if (clean.startsWith(prefix)) {
                      clean = clean.slice(prefix.length).replace(/^\//, '');
                      changed = true;
                  }
              }
          }

          return clean;
      }
          function updateLightbox() {
              const img = lightboxImages[lightboxIndex];
              const url = img.url.replace(/^\//, '');
              const BASE = 'media/georgerieh/T7/photos_from_icloud/';

              const cleanUrl = url.startsWith(BASE) ? url.slice(BASE.length) : url;
              const lightboxImg = document.getElementById('lightbox-img');
          lightboxImg.src = `/thumbnail/${url}?size=800`;
          lightboxImg.style.filter = 'blur(2px)';
          const full = new Image();
          full.onload = () => {
              lightboxImg.src = full.src;
              lightboxImg.style.filter = '';
          };
          full.src = `/files/${url}`;
              document.getElementById('lightbox-location').textContent = 'Date: ' + img.date + ' at ' + (img.city + ', ' + img.country)  || '';
              document.getElementById('lightbox-counter').textContent = `${lightboxIndex + 1} / ${lightboxImages.length}`;
              document.getElementById('lightbox-fav').classList.toggle('active', _favSet ? _favSet.has(cleanUrl) : false);
              const preload = (i) => {
              const p = lightboxImages[i];
              if (!p) return;
              const l = new Image();
              l.src = `/files/${p.url.replace(/^\//, '')}`;
          };
          preload((lightboxIndex + 1) % lightboxImages.length);
          preload((lightboxIndex - 1 + lightboxImages.length) % lightboxImages.length);
          }


          document.getElementById('lightbox-close').addEventListener('click', closeLightbox);
          document.getElementById('lightbox-prev').addEventListener('click', () => { lightboxIndex = (lightboxIndex - 1 + lightboxImages.length) % lightboxImages.length; updateLightbox(); });
          document.getElementById('lightbox-next').addEventListener('click', () => { lightboxIndex = (lightboxIndex + 1) % lightboxImages.length; updateLightbox(); });
          document.getElementById('lightbox-fav').addEventListener('click', () => {
              const url = lightboxImages[lightboxIndex].url.replace(/^\//, '');
              document.getElementById('lightbox-fav').classList.toggle('active', toggleFavorite(url));
          });
          document.getElementById('lightbox').addEventListener('click', e => { if (e.target === document.getElementById('lightbox')) closeLightbox(); });
          document.addEventListener('keydown', e => {
              if (!document.getElementById('lightbox').classList.contains('open')) return;
              if (e.key === 'Escape') closeLightbox();
              if (e.key === 'ArrowLeft') { lightboxIndex = (lightboxIndex - 1 + lightboxImages.length) % lightboxImages.length; updateLightbox(); }
              if (e.key === 'ArrowRight') { lightboxIndex = (lightboxIndex + 1) % lightboxImages.length; updateLightbox(); }
          });

          window.openLightbox = openLightbox;

          // ── MAP ──

          function updateMap(images) {
              initMap();
              leafletMap.on('zoomend moveend', updateHexLayer);
              updateHexLayer();
              if (markerClusters) leafletMap.removeLayer(markerClusters);
              markerClusters = L.markerClusterGroup({ spiderfyOnMaxZoom: true, showCoverageOnHover: false });
              const bounds = [];
              images.forEach((img, i) => {
                  if (!img.lat) return;
                  const url = img.url.replace(/^\//, '');
                  const marker = L.marker([img.lat, img.lon]);
                  marker.bindPopup(`<img src="/thumbnail/${url}" style="width:120px;height:80px;object-fit:cover;border-radius:6px;"><br><small>${img.location||''}</small>`);
                  marker.on('click', () => openLightbox(images, i));
                  markerClusters.addLayer(marker);
                  bounds.push([img.lat, img.lon]);
              });
              leafletMap.addLayer(markerClusters);
              if (bounds.length > 0) leafletMap.fitBounds(bounds, { padding: [20, 20] });
          }

          // ── RENDER ──
          window.currentImages = [];
          document.addEventListener('DOMContentLoaded', () => {
          const interval = setInterval(() => {
              if (window.initMap) {
                  window.initMap();
                  clearInterval(interval);
              }
          }, 100);
          });
                  const observer = new IntersectionObserver((entries) => {
                  entries.forEach(entry => {
                      if (entry.isIntersecting) {
                          const thumb = entry.target;
                          const url = thumb.dataset.thumb;
                          if (url && !thumb.style.backgroundImage) {
                              thumb.style.backgroundImage = `url('${url}')`;
                          }
                          observer.unobserve(thumb);
                      }
          });},{ rootMargin: '200px', threshold: 0 });
          window.renderChunk = function(images) {
          const grid = document.getElementById('photo-grid');
          const favs = _favSet ? [..._favSet] : [];
          const offset = window.currentImages.length;
          window.currentImages.push(...images);
          document.getElementById('empty-state').style.display = 'none';
          images.forEach((img, i) => {
              const url = img.url.replace(/^\//, '');
              const isFav = favs.includes(url);
              const cell = document.createElement('div');
              cell.dataset.url = url;
              cell.className = 'grid-cell';
              cell.draggable = true;
                  cell.addEventListener('dragstart', e => {
                      // if multiple selected, drag all of them
                      const urls = gridSelected.size > 0 ? [...gridSelected] : [url];
                      e.dataTransfer.setData('photo-paths', JSON.stringify(urls));
                      e.dataTransfer.effectAllowed = 'link';
                      // visual feedback
                      cell.style.opacity = '0.5';
                  });
                  cell.addEventListener('dragend', () => {
                      cell.style.opacity = '';
                  });
              cell.innerHTML = `
                  <div class="grid-thumb" data-thumb="/thumbnail/${url}">
                      <div class="grid-overlay">
                          <button class="thumb-btn thumb-fav ${isFav?'fav-active':''}" title="Favorite">♥</button>
                          <button class="thumb-btn thumb-delete" title="Delete">✕</button>
                      </div>
                      <div class="grid-score">${img.score.toFixed(2)}</div>
                  </div>
              `;
                  cell.querySelector('.grid-thumb').addEventListener('click', e => {
                      if (e.target.closest('.thumb-btn')) return;
                      if (gridSelected.size > 0 || e.shiftKey) {
                          if (gridSelected.has(url)) {
                              gridSelected.delete(url);
                              cell.classList.remove('grid-selected');
                          } else {
                              gridSelected.add(url);
                              cell.classList.add('grid-selected');
                          }
                          updateGridToolbar();
                          return;
                      }
                      openLightbox(window.currentImages, offset + i);
                      cell.addEventListener('contextmenu', e => {
                         e.preventDefault();
                      gridSelected.add(url);
                      cell.classList.add('grid-selected');
                      updateGridToolbar();
                  });
                  });
                  cell.querySelector('.thumb-fav').addEventListener('click', e => {
                      e.stopPropagation();
                      const isFav = toggleFavorite(url);
                      e.target.classList.toggle('fav-active', isFav);
                  });
                  cell.querySelector('.thumb-delete').addEventListener('click', async e => {
                      e.stopPropagation();
                      if (!confirm('Delete this photo?')) return;
                      const fd = new FormData();
                      fd.append('image_paths', url);
                      await fetch('/delete_photo', { method: 'POST', body: fd });
                      cell.style.opacity = '0';
                      cell.style.transform = 'scale(0.85)';
                      setTimeout(() => cell.remove(), 300);
                  });
              grid.appendChild(cell);
              requestAnimationFrame(() => {
                  cell.style.transitionDelay = `${Math.min(i * 15, 300)}ms`;
                  cell.classList.add('appear');
              });
          });

          // lazy load new thumbs
          grid.querySelectorAll('.grid-thumb:not([data-observed])').forEach(el => {
              el.dataset.observed = '1';
              observer.observe(el);
          });

          document.getElementById('result-count').textContent = `${window.currentImages.length} photos`;
      };
          window.renderResults = function(data) {
              window.currentImages = data.images;
              const grid = document.getElementById('photo-grid');
              grid.innerHTML = '';
              const favs = getFavorites();

              data.images.forEach((img, i) => {
                  const url = img.url.replace(/^\//, '');
                  const fileUrl = `/files/${url}`;      // full size — for lightbox
                  const thumbUrl = `/thumbnail/${url}`; // small — for grid
                  const isFav = favs.includes(url);

                  const cell = document.createElement('div');
                  cell.dataset.url = url;
                  cell.className = 'grid-cell';
                  cell.innerHTML = `
                      <div class="grid-thumb" data-thumb="/thumbnail/${url}">
                          <div class="grid-overlay">
                              <button class="thumb-btn thumb-fav ${isFav?'fav-active':''}" title="Favorite">♥</button>
                              <button class="thumb-btn thumb-delete" title="Delete">✕</button>
                          </div>
                          <div class="grid-score">${img.score.toFixed(2)}</div>
                      </div>
                  `;
                  cell.querySelector('.grid-thumb').addEventListener('click', e => {
                      if (e.target.closest('.thumb-btn')) return;
                      if (gridSelected.size > 0 || e.shiftKey) {
                          if (gridSelected.has(url)) {
                              gridSelected.delete(url);
                              cell.classList.remove('grid-selected');
                          } else {
                              gridSelected.add(url);
                              cell.classList.add('grid-selected');
                          }
                          updateGridToolbar();
                          return;
                      }
                      openLightbox(data.images, i);
                      cell.addEventListener('contextmenu', e => {
                      e.preventDefault();
                      gridSelected.add(url);
                      cell.classList.add('grid-selected');
                      updateGridToolbar();
                  });
                  });
                  cell.querySelector('.thumb-fav').addEventListener('click', e => {
                      e.stopPropagation();
                      const isFav = toggleFavorite(url);
                      e.target.classList.toggle('fav-active', isFav);
                  });
                  cell.querySelector('.thumb-delete').addEventListener('click', async e => {
                      e.stopPropagation();
                      if (!confirm('Delete this photo?')) return;
                      const fd = new FormData();
                      fd.append('image_paths', url);
                      await fetch('/delete_photo', { method: 'POST', body: fd });
                      cell.style.opacity = '0';
                      cell.style.transform = 'scale(0.85)';
                      setTimeout(() => cell.remove(), 300);
                  });
                  grid.appendChild(cell);
              });

              // staggered appear animation
              requestAnimationFrame(() => {
                  grid.querySelectorAll('.grid-cell').forEach((el, i) => {
                      el.style.transitionDelay = `${Math.min(i * 25, 600)}ms`;
                      el.classList.add('appear');
                  });
              });

              grid.querySelectorAll('.grid-thumb').forEach(el => observer.observe(el));

              document.getElementById('result-count').textContent = `${data.images.length} photos`;
              document.getElementById('toolbar-info').textContent = `${data.images.length} results`;

              updateMap(data.images);

              document.getElementById('empty-state').style.display = 'none';

              // if in gallery view, rebuild
              if (window.currentView === 'gallery') buildGallery(data.images, 0);
          };

          // init favorites strip


          loadFavoritesFromDisk();

          // init with server-rendered data if any
          {% if images %}
          const serverImages = {{ images | tojson }};
          if (serverImages.length > 0) window.renderResults({ images: serverImages });
          {% endif %}
      window.favOverlaySelected = new Set();

      async function openFavoritesOverlay() {
          const overlay = document.getElementById('favorites-overlay');
          const grid = document.getElementById('fav-overlay-grid');
          const countEl = document.getElementById('fav-overlay-count');
          const batchBtn = document.getElementById('fav-batch-delete-btn');

          overlay.style.display = 'flex';
          grid.innerHTML = '';
          window.favOverlaySelected.clear();
          if (batchBtn) batchBtn.style.display = 'none';

          const resp = await fetch('/favorites');
          const { photos } = await resp.json();
          countEl.textContent = `${photos.length} photos`;

          photos.forEach((photo, i) => {
              const url = photo.url.replace(/^\//, '');
              const cell = document.createElement('div');
              cell.className = 'fav-overlay-cell';
              cell.dataset.url = url;
              cell.style.cssText = 'aspect-ratio:1;position:relative;border-radius:4px;overflow:hidden;transition:transform 0.2s, box-shadow 0.2s;';

              cell.innerHTML = `
                  <div class="fav-img-bg" style="width:100%;height:100%;background:url('/thumbnail/${url}') center/cover;cursor:pointer;"></div>
                  <div class="grid-overlay" style="position:absolute;inset:0;background:linear-gradient(160deg,rgba(0,0,0,0.4) 0%,transparent 45%);opacity:0;transition:opacity 0.18s;padding:7px;display:flex;justify-content:space-between;align-items:flex-start;pointer-events:none;">
                      <button class="thumb-btn thumb-fav fav-active" style="pointer-events:auto;" title="Unfavorite">♥</button>
                      <button class="thumb-btn thumb-delete" style="pointer-events:auto;" title="Delete from disk">✕</button>
                  </div>
                  <div class="select-check" style="position:absolute;bottom:8px;right:8px;width:20px;height:20px;border-radius:50%;background:#007aff;color:white;display:none;align-items:center;justify-content:center;font-size:12px;font-weight:bold;box-shadow:0 2px 4px rgba(0,0,0,0.2);">✓</div>
              `;

              // Selection / Lightbox logic
              cell.querySelector('.fav-img-bg').addEventListener('click', (e) => {
                  if (e.shiftKey || window.favOverlaySelected.size > 0) {
                      // Toggle multi-select status
                      if (window.favOverlaySelected.has(url)) {
                          window.favOverlaySelected.delete(url);
                          cell.style.transform = '';
                          cell.style.boxShadow = '';
                          cell.querySelector('.select-check').style.display = 'none';
                      } else {
                          window.favOverlaySelected.add(url);
                          cell.style.transform = 'scale(0.92)';
                          cell.style.boxShadow = '0 0 0 3px #007aff';
                          cell.querySelector('.select-check').style.display = 'flex';
                      }

                      // Show/Hide batch delete button
                      if (batchBtn) {
                          batchBtn.style.display = window.favOverlaySelected.size > 0 ? 'inline-block' : 'none';
                          batchBtn.textContent = `✕ Delete Selected (${window.favOverlaySelected.size}) From Disk`;
                      }
                  } else {
                      // Regular single click goes to Lightbox
                      openLightbox(photos, i);
                  }
              });

              // Hover overlays
              cell.addEventListener('mouseenter', () => cell.querySelector('.grid-overlay').style.opacity = '1');
              cell.addEventListener('mouseleave', () => cell.querySelector('.grid-overlay').style.opacity = '0');

              // Individual quick buttons inside the overlay
              cell.querySelector('.thumb-fav').addEventListener('click', async (e) => {
                  e.stopPropagation();
                  await toggleFavorite(url);
                  cell.style.opacity = '0';
                  setTimeout(() => { cell.remove(); countEl.textContent = `${grid.children.length} photos`; }, 300);
              });

              cell.querySelector('.thumb-delete').addEventListener('click', async (e) => {
                  e.stopPropagation();
                  if (!confirm('Delete this photo from disk permanently?')) return;
                  const fd = new FormData();
                  fd.append('image_paths', url);
                  await fetch('/delete_photo', {method:'POST', body: fd});
                  await toggleFavorite(url);
                  cell.style.opacity = '0';
                  setTimeout(() => { cell.remove(); countEl.textContent = `${grid.children.length} photos`; }, 300);
              });

              grid.appendChild(cell);
          });
      }
      window.openFavoritesOverlay = openFavoritesOverlay;
      async function deleteSelectedFavs() {
          if (window.favOverlaySelected.size === 0) return;

          const count = window.favOverlaySelected.size;
          if (!confirm(`Are you sure you want to permanently delete these ${count} selected photos from your T7 disk drive?`)) return;

          const grid = document.getElementById('fav-overlay-grid');
          const countEl = document.getElementById('fav-overlay-count');
          const batchBtn = document.getElementById('fav-batch-delete-btn');

          document.getElementById('loading-container').style.display = 'block';

          for (const url of [...window.favOverlaySelected]) {
              try {
                  const fd = new FormData();
                  fd.append('image_paths', url);
                  await fetch('/delete_photo', { method: 'POST', body: fd });

                  if (_favSet) _favSet.delete(url);
                  await fetch('/favorites/remove', {
                      method: 'DELETE',
                      headers: {'Content-Type':'application/json'},
                      body: JSON.stringify({path: url})
                  });

                  const cell = grid.querySelector(`[data-url="${CSS.escape(url)}"]`);
                  if (cell) {
                      cell.style.opacity = '0';
                      cell.style.transform = 'scale(0.8)';
                      setTimeout(() => cell.remove(), 250);
                  }

                  window.currentImages = window.currentImages.filter(img => img.url.replace(/^\//, '') !== url);
                  const mainGridCell = document.querySelector(`#photo-grid [data-url="${CSS.escape(url)}"]`);
                  if (mainGridCell) mainGridCell.remove();

              } catch (err) {
                  console.error("Batch deletion processing failure for item:", url, err);
              }
          }

          window.favOverlaySelected.clear();
          if (batchBtn) batchBtn.style.display = 'none';

          setTimeout(() => {
              countEl.textContent = `${grid.children.length} photos`;
              renderFavStrip();
              document.getElementById('loading-container').style.display = 'none';
          }, 300);
      }
      window.deleteSelectedFavs = deleteSelectedFavs;

          document.getElementById('fav-overlay-close').addEventListener('click', () => {
              document.getElementById('favorites-overlay').style.display = 'none';
          });
          let savedFaces = [];
          let selectedFaceEmbedding = null;

          async function loadFaces() {
              const resp = await fetch('/faces');
              const { faces } = await resp.json();
              savedFaces = faces;
              renderFaceSelector();
          }

          function renderFaceSelector() {
              const container = document.getElementById('face-selector');
              const msg = document.getElementById('no-faces-msg');
              container.innerHTML = '';
              if (!savedFaces.length) { msg.style.display = 'block'; return; }
              msg.style.display = 'none';
              savedFaces.forEach(face => {
                  const el = document.createElement('div');
                  el.style.cssText = 'position:relative;cursor:pointer;';
                  el.innerHTML = `
                      <img src="${face.thumbnail}" style="
                          width:44px;height:44px;border-radius:50%;object-fit:cover;
                          border:2px solid transparent;transition:border-color 0.15s;
                      " title="${face.name}">
                      <div style="font-size:9px;text-align:center;color:#8e8e93;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;width:44px;">${face.name}</div>
                  `;
                  const img = el.querySelector('img');
                  el.addEventListener('click', () => {
                      if (selectedFaceEmbedding === face.embedding) {
                          selectedFaceEmbedding = null;
                          img.style.borderColor = 'transparent';
                      } else {
                          selectedFaceEmbedding = face.embedding;
                          // deselect others
                          container.querySelectorAll('img').forEach(i => i.style.borderColor = 'transparent');
                          img.style.borderColor = '#007aff';
                      }
                  });
                  // long press to delete
                  let pressTimer;
                  el.addEventListener('mousedown', () => {
                      pressTimer = setTimeout(async () => {
                          if (!confirm(`Delete face "${face.name}"?`)) return;
                          await fetch(`/faces/${face.id}`, {method:'DELETE'});
                          loadFaces();
                      }, 800);
                  });
                  el.addEventListener('mouseup', () => clearTimeout(pressTimer));
                  container.appendChild(el);
              });
          }

      function showFaceModal(faceEmbedding, croppedCanvas, autoClose = 10000) {
              return new Promise((resolve) => {
                  const timeout = setTimeout(() => {
                      modal.style.display = 'none';
                      resolve();
                  }, autoClose);

                  const done = () => {
                      clearTimeout(timeout);
                      resolve();
                  };
              const modal = document.getElementById('face-modal');
              const canvas = document.getElementById('face-crop-canvas');
              const ctx = canvas.getContext('2d');
              ctx.clearRect(0, 0, 160, 160);
              ctx.drawImage(croppedCanvas, 0, 0, 160, 160);
              modal.style.display = 'flex';

              document.getElementById('face-name-input').value = '';
              document.getElementById('face-name-input').focus();

              document.getElementById('face-modal-save').onclick = async () => {
                  const name = document.getElementById('face-name-input').value.trim();
                  if (!name) { document.getElementById('face-name-input').focus(); return; }
                  const thumbnail = canvas.toDataURL('image/jpeg', 0.8);
                  await fetch('/faces', {
                      method: 'POST',
                      headers: {'Content-Type':'application/json'},
                      body: JSON.stringify({name, embedding: faceEmbedding, thumbnail})
                  });
                  modal.style.display = 'none';
                  loadFaces();
                  done();
              };
              document.getElementById('face-modal-cancel').onclick = () => {
                  modal.style.display = 'none';
                  done();
              };
              document.getElementById('face-name-input').addEventListener('keydown', e => {
                  if (e.key === 'Enter') document.getElementById('face-modal-save').click();
              });
          })
          }

          loadFaces();