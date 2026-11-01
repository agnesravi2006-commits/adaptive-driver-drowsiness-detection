const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const capture = document.createElement("canvas");
const captureCtx = capture.getContext("2d");

const els = {
  connection: document.getElementById("connectionPill"),
  fps: document.getElementById("fpsLabel"),
  cal: document.getElementById("calLabel"),
  banner: document.getElementById("banner"),
  state: document.getElementById("driverState"),
  conf: document.getElementById("confidenceStatus"),
  statusCard: document.getElementById("statusCard"),
  riskFill: document.getElementById("riskFill"),
  riskScore: document.getElementById("riskScore"),
  ear: document.getElementById("earValue"),
  mar: document.getElementById("marValue"),
  closure: document.getElementById("closureValue"),
  blink: document.getElementById("blinkValue"),
  yawn: document.getElementById("yawnValue"),
  nod: document.getElementById("nodValue"),
  perclos: document.getElementById("perclosValue"),
  pitch: document.getElementById("pitchValue"),
  thresholds: document.getElementById("thresholdText"),
  start: document.getElementById("btnStart"),
  stop: document.getElementById("btnStop"),
  calibrate: document.getElementById("btnCalibrate"),
  mute: document.getElementById("btnMute"),
  reset: document.getElementById("btnReset"),
};

let stream = null;
let sending = false;
let muted = false;
let audioCtx = null;
let lastBeep = 0;
let loopId = 0;

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json().catch(() => ({ ok: false, error: "Invalid JSON" }));
  if (!response.ok && !data.error) data.error = `HTTP ${response.status}`;
  return data;
}

function setConnection(text, ok) {
  els.connection.textContent = text;
  els.connection.style.color = ok ? "#3dd68c" : "#e7c04b";
}

function stateClass(state) {
  if (state === "High Risk") return "status-high";
  if (state === "Drowsy") return "status-drowsy";
  if (state === "Low Risk") return "status-low";
  if (state === "Alert") return "status-alert";
  return "status-uncertain";
}

function scoreColor(score, state) {
  if (state === "Face Not Detected" || state === "Detection Uncertain") return "#4cc9f0";
  if (score >= 75) return "#ff4d4f";
  if (score >= 50) return "#f08a3a";
  if (score >= 25) return "#e7c04b";
  return "#3dd68c";
}

function drawLandmarks(payload) {
  const rect = video.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const points = payload.landmarks || [];
  if (!points.length || !payload.image_width) return;
  const sx = canvas.width / payload.image_width;
  const sy = canvas.height / payload.image_height;
  ctx.fillStyle = "#4cc9f0";
  ctx.strokeStyle = "rgba(61, 214, 140, 0.9)";
  ctx.lineWidth = 2;
  for (const point of points) {
    ctx.beginPath();
    ctx.arc(point.x * sx, point.y * sy, 2.4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function beep(repeated) {
  if (muted) return;
  const now = performance.now();
  const gap = repeated ? 280 : 900;
  if (now - lastBeep < gap) return;
  lastBeep = now;
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = "square";
  osc.frequency.value = repeated ? 880 : 520;
  gain.gain.value = 0.05;
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + (repeated ? 0.12 : 0.18));
}

function render(payload) {
  const state = payload.driver_state || "Idle";
  els.state.textContent = state;
  els.conf.textContent = `Confidence: ${payload.confidence_status || "waiting"} (${payload.confidence ?? 0})`;
  els.statusCard.className = `status-card ${stateClass(state)}`;
  els.riskScore.textContent = payload.risk_score ?? 0;
  els.riskFill.style.width = `${Math.min(payload.risk_score || 0, 100)}%`;
  els.riskFill.style.background = scoreColor(payload.risk_score || 0, state);
  els.ear.textContent = payload.ear ?? "—";
  els.mar.textContent = payload.mar ?? "—";
  els.closure.textContent = `${(payload.eye_closure_duration_s ?? 0).toFixed(2)} s`;
  els.blink.textContent = payload.blink_count ?? 0;
  els.yawn.textContent = payload.yawn_count ?? 0;
  els.nod.textContent = payload.nod_count ?? 0;
  els.perclos.textContent = payload.perclos ?? 0;
  els.pitch.textContent = payload.head_pitch ?? "—";
  els.fps.textContent = `FPS ${payload.fps ?? "—"}`;
  const cal = payload.calibration || {};
  els.cal.textContent = cal.calibrated
    ? `Calibrated · EAR close ${cal.ear_close_threshold}`
    : payload.calibrating
      ? `Calibrating ${(cal.progress * 100).toFixed(0)}%`
      : "Not calibrated";
  els.thresholds.textContent = cal.calibrated
    ? `Baseline EAR ${cal.baseline_ear} (σ ${cal.baseline_ear_std}). Close threshold ${cal.ear_close_threshold}. Yawn MAR ${cal.mar_yawn_threshold}.`
    : "Start calibration while sitting naturally in view of the camera.";

  const banner = payload.banner || payload.error || "";
  if (banner) {
    els.banner.textContent = banner;
    els.banner.classList.remove("hidden");
    els.banner.className = "banner";
    if (payload.alert_level >= 3) els.banner.classList.add("level-3");
    else if (payload.alert_level === 2) els.banner.classList.add("level-2");
    else if (payload.alert_level === 1) els.banner.classList.add("level-1");
    else els.banner.classList.add("uncertain");
  } else {
    els.banner.classList.add("hidden");
  }

  if (payload.sound_alert) beep(Boolean(payload.repeated_sound));
  drawLandmarks(payload);
}

async function sendFrame() {
  if (sending || video.readyState < 2) return;
  sending = true;
  capture.width = 640;
  capture.height = Math.max(360, Math.round((video.videoHeight / video.videoWidth) * 640) || 480);
  captureCtx.drawImage(video, 0, 0, capture.width, capture.height);
  const image = capture.toDataURL("image/jpeg", 0.62);
  try {
    const payload = await api("/api/analyze", { method: "POST", body: JSON.stringify({ image }) });
    render(payload);
    if (payload.ok === false && payload.error) setConnection(payload.error, false);
    else setConnection("Live", true);
  } catch (err) {
    setConnection("Backend unreachable", false);
  } finally {
    sending = false;
  }
}

function loop() {
  sendFrame();
  loopId = window.setTimeout(loop, 120);
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: 640, height: 480 }, audio: false });
  video.srcObject = stream;
  await video.play();
}

function stopCamera() {
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

els.start.addEventListener("click", async () => {
  try {
    await startCamera();
    await api("/api/start", { method: "POST", body: "{}" });
    window.clearTimeout(loopId);
    loop();
  } catch (err) {
    setConnection("Webcam permission denied", false);
    els.banner.textContent = "Allow camera access in the browser, then click Start Monitoring again.";
    els.banner.classList.remove("hidden");
  }
});

els.stop.addEventListener("click", async () => {
  window.clearTimeout(loopId);
  stopCamera();
  await api("/api/stop", { method: "POST", body: "{}" });
  setConnection("Stopped", true);
});

els.calibrate.addEventListener("click", async () => {
  if (!stream) {
    await startCamera();
    window.clearTimeout(loopId);
    loop();
  }
  const payload = await api("/api/calibrate/start", { method: "POST", body: "{}" });
  render(payload);
});

els.mute.addEventListener("click", async () => {
  muted = !muted;
  els.mute.textContent = muted ? "Unmute Alerts" : "Mute Alerts";
  await api("/api/mute", { method: "POST", body: JSON.stringify({ muted }) });
});

els.reset.addEventListener("click", async () => {
  const payload = await api("/api/reset", { method: "POST", body: "{}" });
  render(payload);
});

api("/api/health")
  .then((data) => setConnection(data.analyzer_available ? "Backend ready" : "Backend ready, MediaPipe missing", data.ok))
  .catch(() => setConnection("Backend unreachable", false));
