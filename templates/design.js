document
          .getElementById("memory-today-btn")
          .addEventListener("click", async () => {
            document.getElementById("memory-loading").style.display = "flex";
            document.getElementById("memory-content").style.display = "none";

            const resp = await fetch("/photo_of_day");
            const data = await resp.json();
            currentMemory = data;

            document.getElementById("memory-title").textContent = data.title;
            document.getElementById("memory-subtitle").textContent =
              data.subtitle;

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
              cell.style.cssText =
                "aspect-ratio:1;border-radius:4px;cursor:pointer;";
              cell.style.background = `url('/thumbnail/${url}') center/cover`;
              cell.addEventListener(
                "click",
                () =>
                  window.openLightbox &&
                  window.openLightbox(data.photos, i + 1),
              );
              grid.appendChild(cell);
            });

            document.getElementById("memory-loading").style.display = "none";
            document.getElementById("memory-content").style.display = "block";
          });
        const gridSelected = new Set();
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
          renderMiniTable(
            "bottom-countries-table",
            data.bottom_countries,
            "bottom",
          );
        }

        function renderMonthChart(byMonth) {
          const el = document.getElementById("month-chart");
          const valid = byMonth.filter((d) => d.month && d.month.includes(":"));
          if (!valid.length) {
            el.innerHTML =
              '<span style="color:#8e8e93;font-size:12px">No data</span>';
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
          const valid = byDow.filter(
            (d) => d.dow !== null && d.dow !== undefined,
          );
          if (!valid.length) {
            el.innerHTML =
              '<span style="color:#8e8e93;font-size:12px">No data</span>';
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
      </script>
    </div>
    <!-- end info-pane -->

    <!-- LIGHTBOX -->
    <div
      id="album-overlay"
      style="
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(242, 242, 247, 0.97);
        backdrop-filter: blur(20px);
        z-index: 500;
        flex-direction: column;
        overflow: hidden;
      "
    >
      <div
        style="
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px 16px;
          background: rgba(255, 255, 255, 0.88);
          border-bottom: 1px solid rgba(0, 0, 0, 0.08);
        "
      >
        <button
          id="album-back"
          style="
            border: none;
            background: none;
            font-size: 20px;
            cursor: pointer;
            color: #007aff;
            padding: 4px 8px;
          "
        >
          ‹
        </button>
        <div
          id="album-overlay-title"
          style="font-size: 17px; font-weight: 600; flex: 1"
        ></div>
        <div
          id="album-overlay-count"
          style="font-size: 12px; color: #8e8e93"
        ></div>
      </div>
      <div
        id="album-overlay-grid"
        style="
          flex: 1;
          overflow-y: auto;
          padding: 12px;
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
          gap: 3px;
          align-content: start;
        "
      ></div>
    </div>
    <div
      id="favorites-overlay"
      style="
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(242, 242, 247, 0.97);
        backdrop-filter: blur(20px);
        z-index: 500;
        flex-direction: column;
        overflow: hidden;
      "
    >
      <div
        style="
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px 16px;
          background: rgba(255, 255, 255, 0.88);
          border-bottom: 1px solid rgba(0, 0, 0, 0.08);
        "
      >
        <button
          id="fav-overlay-close"
          style="
            border: none;
            background: none;
            font-size: 20px;
            cursor: pointer;
            color: #007aff;
            padding: 4px 8px;
          "
        >
          ‹
        </button>
        <button
          id="fav-batch-delete-btn"
          onclick="window.deleteSelectedFavs()"
          style="
            display: none;
            color: #ff3b30;
            margin-left: 15px;
            background: none;
            border: none;
            font-weight: 600;
            cursor: pointer;
          "
        >
          ✕ Delete Selected From Disk
        </button>
        <div style="font-size: 17px; font-weight: 600; flex: 1">Favorites</div>
        <div
          id="fav-overlay-count"
          style="font-size: 12px; color: #8e8e93"
        ></div>
      </div>
      <div
        id="fav-overlay-grid"
        style="
          flex: 1;
          overflow-y: auto;
          padding: 12px;
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
          gap: 3px;
          align-content: start;
        "
      ></div>
    </div>
    <div
      id="face-upload-modal"
      style="
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.7);
        z-index: 3000;
        align-items: center;
        justify-content: center;
        backdrop-filter: blur(10px);
      "
    >
      <div
        style="
          background: white;
          border-radius: 20px;
          padding: 24px;
          width: 320px;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 16px;
          box-shadow: 0 40px 100px rgba(0, 0, 0, 0.4);
          position: relative;
        "
      >
        <button
          id="face-upload-close"
          style="
            position: absolute;
            top: 12px;
            right: 12px;
            background: none;
            border: none;
            font-size: 18px;
            cursor: pointer;
            color: #8e8e93;
            transition: color 0.15s;
          "
        >
          ✕
        </button>
        <div style="font-size: 17px; font-weight: 600">Extract a Face</div>

        <div
          id="face-drop-zone"
          style="
            width: 100%;
            height: 140px;
            border: 2px dashed #c7c7cc;
            border-radius: 12px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #8e8e93;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
            padding: 10px;
            background: #fafafa;
          "
        >
          <span style="font-size: 28px; margin-bottom: 8px"></span>
          Drop a picture here<br />or click to browse
          <input
            type="file"
            id="face-file-input"
            accept="image/*"
            style="display: none"
          />
        </div>

        <div
          id="face-upload-loading"
          style="
            display: none;
            font-size: 13px;
            color: #007aff;
            font-weight: 500;
          "
        >
          <div
            class="spinner"
            style="
              width: 20px;
              height: 20px;
              margin: 0 auto 10px;
              border-width: 2px;
            "
          ></div>
          Detecting face...
        </div>
      </div>
    </div>
    <div
      id="face-modal"
      style="
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.7);
        z-index: 3000;
        align-items: center;
        justify-content: center;
        backdrop-filter: blur(10px);
      "
    >
      <div
        style="
          background: white;
          border-radius: 20px;
          padding: 24px;
          width: 320px;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 16px;
          box-shadow: 0 40px 100px rgba(0, 0, 0, 0.4);
        "
      >
        <div style="font-size: 17px; font-weight: 600">Face detected</div>
        <canvas
          id="face-crop-canvas"
          width="160"
          height="160"
          style="border-radius: 50%; border: 3px solid #007aff"
        ></canvas>
        <div style="font-size: 13px; color: #8e8e93; text-align: center">
          Save this face to search for similar photos?
        </div>
        <input
          id="face-name-input"
          placeholder="Enter a name..."
          style="
            width: 100%;
            padding: 10px 12px;
            border: 1px solid #e5e5ea;
            border-radius: 10px;
            font-size: 14px;
            font-family: inherit;
            outline: none;
          "
        />
        <div style="display: flex; gap: 8px; width: 100%">
          <button
            id="face-modal-cancel"
            style="
              flex: 1;
              padding: 11px;
              border: 1px solid #e5e5ea;
              background: white;
              border-radius: 10px;
              font-size: 14px;
              font-weight: 600;
              cursor: pointer;
              font-family: inherit;
            "
          >
            Skip
          </button>
          <button
            id="face-modal-save"
            style="
              flex: 1;
              padding: 11px;
              border: none;
              background: #007aff;
              color: white;
              border-radius: 10px;
              font-size: 14px;
              font-weight: 600;
              cursor: pointer;
              font-family: inherit;
            "
          >
            Save face
          </button>
        </div>
      </div>
    </div>
    <div id="lightbox">
      <button class="lb-nav" id="lightbox-prev">‹</button>
      <div class="lightbox-content">
        <img id="lightbox-img" src="" alt="" />
        <div class="lightbox-meta">
          <span id="lightbox-counter"></span>
          <span id="lightbox-location"></span>
        </div>
      </div>
      <button class="lb-nav" id="lightbox-next">›</button>
      <div class="lightbox-actions">
        <button class="lb-action" id="lightbox-fav">♥ Favorite</button>
        <button class="lb-action" id="lightbox-close">✕</button>
      </div>
    </div>

    <!-- LOADING -->
    <div id="loading-container">
      <div id="loading-text">Loading...</div>
      <div class="loading-track"><div id="loading-bar"></div></div>
    </div>

    <script>
      window.currentImages = [];
      window.currentView = "grid";

      window.setView = function (view) {
        window.currentView = view;
        ["grid", "gallery", "info", "memory"].forEach((v) =>
          document
            .getElementById(`btn-${v}`)
            ?.classList.toggle("active", v === view),
        );

        const hasResults = window.currentImages.length > 0;

        document.getElementById("results-area").style.display =
          view === "grid" ? "flex" : "none";
        document.getElementById("map-pane").style.display =
          view === "grid" ? "flex" : "none";
        document.getElementById("empty-state").style.display =
          view === "grid" && !hasResults ? "flex" : "none";
        document
          .getElementById("gallery-pane")
          .classList.toggle("active", view === "gallery");
        document.getElementById("info-pane").style.display =
          view === "info" ? "flex" : "none";
        document.getElementById("memory-pane").style.display =
          view === "memory" ? "flex" : "none";
        document.getElementById("review-pane").style.display =
          view === "review" ? "flex" : "none";
        document.querySelector(".sidebar").style.display =
          view === "info" ? "none" : "";
        document.querySelector(".main").style.marginLeft =
          view === "info" ? "0" : "260px";

        if (view === "gallery" && hasResults)
          buildGallery(window.currentImages, 0);
        if (view === "info" && !window.statsLoaded) {
          window.statsLoaded = true;
          loadStats();
        }
        if (view === "info") {
          document.getElementById("info-pane").scrollTop = 0;
        }
        if (view === "memory") loadMemory();
        if (
          view === "review" &&
          !document.getElementById("review-grid").children.length
        )
          loadReviewBatch();
      };
      let galleryIndex = 0;

      function buildGallery(images, startIndex) {
        const main = document.getElementById("gallery-main");
        const strip = document.getElementById("gallery-strip");
        main.querySelectorAll(".gallery-photo").forEach((el) => el.remove());
        strip.innerHTML = "";
        galleryIndex = startIndex;

        images.forEach((img, i) => {
          const url = img.url.replace(/^\//, "");
          const photo = document.createElement("div");
          photo.className =
            "gallery-photo" + (i === startIndex ? " active" : "");
          photo.style.backgroundImage = `url('/thumbnail/${url}')`;
          main.insertBefore(photo, main.querySelector(".gallery-controls"));

          const thumb = document.createElement("div");
          thumb.className = "strip-thumb" + (i === startIndex ? " active" : "");
          thumb.style.backgroundImage = `url('/thumbnail/${url}')`;
          photo.style.cursor = "pointer";
          photo.addEventListener("click", () => window.openLightbox(images, i));
          thumb.addEventListener("click", () => setGalleryIndex(i));
          strip.appendChild(thumb);
        });
      }

      function setGalleryIndex(i) {
        document
          .querySelectorAll(".gallery-photo")
          .forEach((p, idx) => p.classList.toggle("active", idx === i));
        document
          .querySelectorAll(".strip-thumb")
          .forEach((t, idx) => t.classList.toggle("active", idx === i));
        galleryIndex = i;
        document
          .querySelectorAll(".strip-thumb")
          [
            i
          ]?.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
      }

      document.getElementById("gallery-prev").addEventListener("click", () => {
        const len = window.currentImages.length;
        if (len) setGalleryIndex((galleryIndex - 1 + len) % len);
      });
      document.getElementById("gallery-next").addEventListener("click", () => {
        const len = window.currentImages.length;
        if (len) setGalleryIndex((galleryIndex + 1) % len);
      });
      document.getElementById("image-in").addEventListener("change", (e) => {
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
        const clampedWidth = Math.max(
          200,
          Math.min(newWidth, splitRect.width * 0.7),
        );
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

        // ========================================================
        // 1. RENDER THE ACTIVE "CURRENT MEMORY" SUB-MODULE BOX
        // ========================================================
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
          hero.onclick = () =>
            window.openLightbox && window.openLightbox(photos, 0);
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

        const existingToolbar = document.getElementById(
          "memory-select-toolbar",
        );
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
            if (
              !confirm(`Remove ${selected.size} photo(s) from this collection?`)
            )
              return;
            for (const url of selected) {
              const cell = grid.querySelector(`[data-url="${url}"]`);
              if (cell) {
                cell.style.opacity = "0";
                cell.style.transform = "scale(0.85)";
                cell.style.transition = "all 0.2s";
              }
              await fetch(
                `/collections/${encodeURIComponent(col.name)}/remove`,
                {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ path: url }),
                },
              );
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
          document.getElementById("memory-subtitle").textContent =
            data.subtitle;

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
                  !confirm(
                    `Permanently delete ${selected.size} photo(s) from disk?`,
                  )
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
                  if (window.openLightbox)
                    window.openLightbox(data.photos, i + 1);
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

      document
        .getElementById("memory-delete-btn")
        .addEventListener("click", () => {
          loadMemory(); // dismiss = load a new one
        });

      document
        .getElementById("memory-refresh-btn")
        .addEventListener("click", () => {
          loadMemory();
        });

      // ── AUTO-LOAD on new session ──
      function checkAndLoadMemory() {
        const lastVisit = localStorage.getItem("mem_last_visit");
        const today = new Date().toDateString();
        localStorage.setItem("mem_last_visit", today);
        // load if first visit today
        if (lastVisit !== today) {
          // auto-switch to memory tab on new day
          setTimeout(() => setView("memory"), 1200); // after models load
        }
      }

      checkAndLoadMemory();
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
          el.querySelector(".album-del").addEventListener(
            "click",
            async (e) => {
              e.stopPropagation();
              if (!confirm(`Delete album "${album.name}"?`)) return;
              await fetch(`/albums/${encodeURIComponent(album.name)}`, {
                method: "DELETE",
              });
              loadAlbums();
            },
          );

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
          el.addEventListener(
            "dragleave",
            () => (el.style.background = "white"),
          );
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
            el.querySelector(".album-name").textContent +=
              ` (+${paths.length})`;
            setTimeout(() => {
              el.style.background = "white";
              loadAlbums();
            }, 600);
          });

          list.appendChild(el);
        });
      }

      // new album button
      document
        .getElementById("new-album-btn")
        .addEventListener("click", async () => {
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
            .addEventListener("click", () =>
              window.openLightbox(data.photos, i),
            );
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
            .classList.toggle(
              "fav-active",
              window.getFavorites().includes(url),
            );

          cell
            .querySelector(".thumb-delete")
            .addEventListener("click", async (e) => {
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
          return new Set(
            JSON.parse(localStorage.getItem("review_seen") || "[]"),
          );
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
        document.getElementById("review-progress").textContent =
          `${total} reviewed`;

        const grid = document.getElementById("review-grid");
        grid.innerHTML = "";

        data.photos.forEach((photo, i) => {
          const url = photo.url.replace(/^\//, "");
          const cell = document.createElement("div");
          cell.style.cssText =
            "aspect-ratio:1;position:relative;border-radius:4px;overflow:hidden;opacity:0;transform:scale(0.94);transition:opacity 0.3s ease,transform 0.3s ease;";
          cell.innerHTML = `
            <div style="width:100%;height:100%;background:url('/thumbnail/${url}') center/cover;cursor:pointer;"></div>
            <div style="position:absolute;inset:0;background:linear-gradient(160deg,rgba(0,0,0,0.35) 0%,transparent 45%);opacity:0;transition:opacity 0.18s;padding:7px;display:flex;justify-content:flex-end;align-items:flex-start;">
                <button class="thumb-btn thumb-delete" title="Delete">✕</button>
            </div>
        `;

          cell.querySelector("div").addEventListener("click", (e) => {
            if (e.target.closest(".thumb-btn")) return;
            window.openLightbox(reviewPhotos, i);
          });

          const overlay = cell.querySelector("div + div");
          cell.addEventListener(
            "mouseenter",
            () => (overlay.style.opacity = "1"),
          );
          cell.addEventListener(
            "mouseleave",
            () => (overlay.style.opacity = "0"),
          );

          cell
            .querySelector(".thumb-delete")
            .addEventListener("click", async (e) => {
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
      document
        .getElementById("review-reset-btn")
        .addEventListener("click", () => {
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

      document
        .getElementById("grid-fav-selected")
        .addEventListener("click", () => {
          gridSelected.forEach((url) => window.toggleFavorite(url));
          document.getElementById("grid-fav-selected").textContent = "✓ Done";
          setTimeout(() => {
            document.getElementById("grid-fav-selected").textContent =
              "♥ Favorite";
            clearGridSelection();
          }, 800);
        });
      document.getElementById("clear-btn").addEventListener("click", () => {
        document.getElementById("start-date").value = "";
        document.getElementById("end-date").value = "";
        document.getElementById("country-filter").value = "";
        document.getElementById("city-filter").value = "";
        document.getElementById("h3-filter").value = "";
        document.getElementById("limit").value = "";
        document.getElementById("image-in").value = "";
        document.getElementById("file-label-text").textContent =
          "Search by photo";
        document.getElementById("ref-preview").style.display = "none";
        // clear selected face
        selectedFaceEmbedding = null;
        document
          .querySelectorAll("#face-selector img")
          .forEach((i) => (i.style.borderColor = "transparent"));
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
        document.getElementById("file-label-text").textContent =
          "Search by photo";
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

            const cell = document.querySelector(
              `[data-url="${CSS.escape(url)}"]`,
            );
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
        if (typeof renderFavStrip === "function") renderFavStrip();
        if (typeof updateGridToolbar === "function") updateGridToolbar();

        document.getElementById("result-count").textContent =
          `${window.currentImages.length} photos`;
        document.getElementById("loading-container").style.display = "none";
      }

      // Expose it to your global HTML window context
      window.deleteSelectedPhotos = deleteSelectedPhotos;
      document
        .getElementById("grid-delete-selected")
        .addEventListener("click", async () => {
          if (!confirm(`Permanently delete ${gridSelected.size} photo(s)?`))
            return;
          for (const url of gridSelected) {
            const fd = new FormData();
            fd.append("image_paths", url);
            await fetch("/delete_photo", { method: "POST", body: fd });
            const cell = document.querySelector(
              `.grid-cell[data-url="${url}"]`,
            );
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