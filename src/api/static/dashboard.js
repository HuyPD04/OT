const CLASS_NAMES = {
  0: "person",
  2: "car",
  3: "motorcycle",
  5: "bus",
  7: "truck",
};

const preview = document.querySelector("#preview");
const previewFrame = document.querySelector(".preview-frame");
const previewEmpty = document.querySelector("#preview-empty");
const connection = document.querySelector("#connection");
const previewLabel = document.querySelector("#preview-label");
let previewMode = "annotated";

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", "\"": "&quot;",
  })[character]);
}

function className(classId) {
  return CLASS_NAMES[classId] ?? (classId ?? "—");
}

function setConnection(state, text) {
  connection.className = `connection ${state}`;
  connection.textContent = text;
}

function setPreview(mode) {
  previewMode = mode;
  const path = mode === "raw" ? "/api/v1/preview/raw.mjpeg" : "/api/v1/preview/annotated.mjpeg";
  previewLabel.textContent = mode === "raw" ? "Camera gốc · độ trễ thấp" : "Overlay mới nhất";
  previewFrame.classList.remove("ready");
  previewEmpty.textContent = "Đang kết nối luồng camera...";
  preview.src = `${path}?started=${Date.now()}`;
  document.querySelectorAll("[data-preview-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.previewMode === mode);
  });
}

preview.addEventListener("load", () => previewFrame.classList.add("ready"));
preview.addEventListener("error", () => {
  previewFrame.classList.remove("ready");
  previewEmpty.textContent = "Không thể kết nối preview";
});
document.querySelectorAll("[data-preview-mode]").forEach((button) => {
  button.addEventListener("click", () => setPreview(button.dataset.previewMode));
});

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  const body = await response.json().catch(() => null);
  if (!response.ok && response.status !== 404 && response.status !== 503) {
    throw new Error(`${path}: HTTP ${response.status}`);
  }
  return { status: response.status, body };
}

function updateSummary(status) {
  const cards = [
    ["detection", "detection-version", "detection-frame"],
    ["tracks", "track-version", "track-frame"],
    ["plates", "plate-version", "plate-frame"],
  ];
  cards.forEach(([key, versionId, frameId]) => {
    const packet = status?.[key];
    document.querySelector(`#${versionId}`).textContent = packet ? packet.version : "—";
    document.querySelector(`#${frameId}`).textContent = packet?.frame_id
      ? `Frame ${packet.frame_id}`
      : "Chưa có frame";
  });
}

function updateReadiness(readiness) {
  const ready = readiness?.body?.ready === true;
  document.querySelector("#readiness").textContent = ready ? "READY" : "WAITING";
  document.querySelector("#readiness-detail").textContent = ready
    ? "Pipeline sẵn sàng"
    : "Đang chờ worker hoặc camera";
}

function updateWorkers(workers = {}) {
  const container = document.querySelector("#workers");
  const entries = Object.entries(workers);
  if (!entries.length) {
    container.innerHTML = '<div class="muted">Chưa có trạng thái worker</div>';
    return;
  }
  container.innerHTML = entries.map(([key, worker]) => `
    <article class="worker">
      <span class="worker-name">${escapeHtml(key)}</span>
      <span class="worker-status ${escapeHtml(worker.status)}">${escapeHtml(worker.status)}</span>
      <span class="worker-message" title="${escapeHtml(worker.message)}">${escapeHtml(worker.message || "—")}</span>
    </article>
  `).join("");
}

function setTable(tableId, metaId, packet, kind) {
  const body = document.querySelector(`#${tableId}`);
  const meta = document.querySelector(`#${metaId}`);
  if (!packet) {
    meta.textContent = "Chưa có dữ liệu";
    body.innerHTML = '<tr><td colspan="4" class="empty-cell">Đang chờ kết quả</td></tr>';
    return;
  }

  const items = kind === "track" ? packet.tracks : packet.plates;
  const duration = kind === "track" ? packet.tracking_ms : packet.inference_ms;
  meta.textContent = `Frame ${packet.frame.frame_id} · ${duration.toFixed(1)} ms`;
  if (!items.length) {
    body.innerHTML = '<tr><td colspan="4" class="empty-cell">Không có object</td></tr>';
    return;
  }
  body.innerHTML = items.map((item) => {
    const id = kind === "track" ? item.track_id : item.track_id;
    return `<tr>
      <td>${escapeHtml(id)}</td>
      <td>${escapeHtml(className(item.class_id))}</td>
      <td>${item.confidence == null ? "—" : Number(item.confidence).toFixed(2)}</td>
      <td>${item.x1}, ${item.y1}, ${item.x2}, ${item.y2}</td>
    </tr>`;
  }).join("");
}

async function refreshDashboard() {
  try {
    const [ready, status, tracks, plates] = await Promise.all([
      fetchJson("/health/ready"),
      fetchJson("/api/v1/status"),
      fetchJson("/api/v1/tracks/latest"),
      fetchJson("/api/v1/plates/latest"),
    ]);
    setConnection("online", "API kết nối");
    updateReadiness(ready);
    updateSummary(status.body);
    updateWorkers(status.body?.workers ?? ready.body?.workers);
    setTable("tracks-body", "tracks-meta", tracks.status === 200 ? tracks.body : null, "track");
    setTable("plates-body", "plates-meta", plates.status === 200 ? plates.body : null, "plate");
  } catch (error) {
    setConnection("offline", "API không phản hồi");
    console.warn("Dashboard refresh failed", error);
  }
}

setPreview(previewMode);
refreshDashboard();
setInterval(refreshDashboard, 1000);
