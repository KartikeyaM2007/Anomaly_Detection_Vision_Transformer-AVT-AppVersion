const els = {
  runtimeStatus: document.getElementById("runtimeStatus"),
  cameraState: document.getElementById("cameraState"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  resetBtn: document.getElementById("resetBtn"),
  camera: document.getElementById("camera"),
  capture: document.getElementById("capture"),
  videoShell: document.getElementById("videoShell"),
  verdict: document.getElementById("verdict"),
  threatScore: document.getElementById("threatScore"),
  confidenceScore: document.getElementById("confidenceScore"),
  featureCount: document.getElementById("featureCount"),
  events: document.getElementById("events"),
  device: document.getElementById("device"),
  checkpoint: document.getElementById("checkpoint"),
  threshold: document.getElementById("threshold"),
  thresholdInput: document.getElementById("thresholdInput"),
  thresholdValue: document.getElementById("thresholdValue"),
  liveThresholdInput: document.getElementById("liveThresholdInput"),
  liveThresholdValue: document.getElementById("liveThresholdValue"),
  screenFocusInput: document.getElementById("screenFocusInput"),
  uploadForm: document.getElementById("uploadForm"),
  videoFile: document.getElementById("videoFile"),
  uploadResult: document.getElementById("uploadResult"),
  timeline: document.getElementById("timeline"),
};

let stream = null;
let timer = null;
let busy = false;

function pct(value) {
  return `${Math.round(value * 100)}%`;
}

function pct1(value) {
  return `${(value * 100).toFixed(1)}%`;
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

async function loadHealth() {
  const health = await api("/api/health");
  els.runtimeStatus.textContent = health.ready ? "Runtime ready" : "Runtime not ready";
  els.runtimeStatus.className = `status ${health.ready ? "ok" : "bad"}`;
  els.device.textContent = health.device || "--";
  els.checkpoint.textContent = health.checkpoint_exists ? "best_model.pt found" : "missing";
  els.threshold.textContent = health.threshold;
  els.thresholdInput.value = health.threshold;
  els.thresholdValue.textContent = pct(health.threshold);
  els.liveThresholdInput.value = health.live_threshold || 0.12;
  els.liveThresholdValue.textContent = pct(health.live_threshold || 0.12);
  if (health.error) {
    els.cameraState.textContent = health.error;
  }
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  els.camera.srcObject = stream;
  els.startBtn.disabled = true;
  els.stopBtn.disabled = false;
  els.cameraState.textContent = "Scoring live frames";
  timer = setInterval(sendFrame, 250);
}

function stopCamera() {
  if (timer) clearInterval(timer);
  timer = null;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  els.camera.srcObject = null;
  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
  els.cameraState.textContent = "Camera idle";
}

async function sendFrame() {
  if (busy || !stream) return;
  busy = true;
  try {
    const ctx = els.capture.getContext("2d");
    ctx.drawImage(els.camera, 0, 0, els.capture.width, els.capture.height);
    const image = els.capture.toDataURL("image/jpeg", 0.8);
    const data = await api("/api/live-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image,
        threshold: Number(els.liveThresholdInput.value),
        focusScreen: els.screenFocusInput.checked,
      }),
    });
    renderLive(data);
  } catch (err) {
    els.cameraState.textContent = err.message;
  } finally {
    busy = false;
  }
}

function renderLive(data) {
  if (!data.ready) {
    els.verdict.textContent = "Warming";
    els.cameraState.textContent = data.needed_frames
      ? `Collecting frames: ${data.needed_frames} more`
      : data.error || "Runtime not ready";
    return;
  }
  const result = data.result;
  const isAlert = result.prediction === "ANOMALY";
  els.videoShell.classList.toggle("alert", isAlert);
  els.verdict.textContent = isAlert ? "Threat" : "Normal";
  els.threatScore.textContent = pct1(result.prob_anomaly);
  els.confidenceScore.textContent = pct(result.confidence);
  els.featureCount.textContent = data.feature_count;
  const basis = result.basis ? `, ${result.basis}` : "";
  const focus = els.screenFocusInput.checked ? ", screen focus" : "";
  const state = result.alert_state ? `, ${result.alert_state}` : "";
  els.cameraState.textContent = `Last score: ${new Date().toLocaleTimeString()}${basis}${state}${focus}`;
  renderEvents(data.events || []);
}

function renderEvents(events) {
  if (!events.length) {
    els.events.className = "events empty";
    els.events.textContent = "No alerts";
    return;
  }
  els.events.className = "events";
  els.events.innerHTML = events
    .map((event) => `<div><strong>${event.time}</strong><span>${pct(event.probability)} threat</span></div>`)
    .join("");
}

async function reset() {
  await api("/api/reset", { method: "POST" });
  els.featureCount.textContent = "0";
  els.threatScore.textContent = "--";
  els.confidenceScore.textContent = "--";
  els.verdict.textContent = "Idle";
  els.videoShell.classList.remove("alert");
  renderEvents([]);
}

async function analyzeVideo(event) {
  event.preventDefault();
  const file = els.videoFile.files[0];
  if (!file) return;
  els.uploadResult.textContent = `Analyzing video on ${els.device.textContent || "runtime"}...`;
  els.timeline.innerHTML = "";

  const form = new FormData();
  form.append("video", file);
  form.append("threshold", els.thresholdInput.value);
  try {
    const data = await api("/api/analyze-video", { method: "POST", body: form });
    renderUpload(data);
  } catch (err) {
    els.uploadResult.textContent = err.message;
  }
}

function renderUpload(data) {
  const overall = data.operational || data.overall;
  const raw = data.overall;
  const peak = data.peak_segment;
  els.uploadResult.innerHTML = `
    <strong>${overall.prediction}</strong>
    <span>${pct1(overall.prob_anomaly)} threat probability</span>
    <span>raw model ${pct1(raw.prob_anomaly)}</span>
    <span>peak ${pct1(data.peak_score)}</span>
    <span>${data.clips} clips, ${data.duration.toFixed(1)}s</span>
    ${peak ? `<span>peak at ${peak.start}s-${peak.end}s</span>` : ""}
  `;
  const duration = Math.max(data.duration, 0.1);
  els.timeline.innerHTML = data.timeline
    .map((seg) => {
      const left = (seg.start / duration) * 100;
      const width = Math.max(((seg.end - seg.start) / duration) * 100, 1);
      const alert = seg.prediction === "ANOMALY";
      return `<div class="${alert ? "danger" : "clear"}" style="left:${left}%;width:${width}%;" title="${seg.start}s-${seg.end}s ${pct(seg.prob_anomaly)}"></div>`;
    })
    .join("");
}

els.startBtn.addEventListener("click", startCamera);
els.stopBtn.addEventListener("click", stopCamera);
els.resetBtn.addEventListener("click", reset);
els.uploadForm.addEventListener("submit", analyzeVideo);
els.thresholdInput.addEventListener("input", () => {
  els.thresholdValue.textContent = pct(Number(els.thresholdInput.value));
});
els.liveThresholdInput.addEventListener("input", () => {
  els.liveThresholdValue.textContent = pct(Number(els.liveThresholdInput.value));
});
loadHealth().catch((err) => {
  els.runtimeStatus.textContent = err.message;
  els.runtimeStatus.className = "status bad";
});
