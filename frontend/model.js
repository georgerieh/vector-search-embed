import { InferenceSession, Tensor } from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.min.mjs';
          import { FaceDetector, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs';

          const loadingBar = document.getElementById('loading-bar');
          const loadingText = document.getElementById('loading-text');
          const btn = document.querySelector('button[name="forwardBtn"]');
          btn.disabled = true;
          let dinoSession = null;
          
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
      