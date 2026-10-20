"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  kind: "text",
  status: null,
  file: null,
  memo: null,
  recorder: null,
  jobId: null,
  source: null,
  base: "",
  started: 0,
  timer: null,
  hosts: ["A", "B"],
};

const LANGUAGE_VOICE_GROUP = {
  en: "American English",
  "en-gb": "British English",
  es: "Spanish",
  fr: "French",
  hi: "Hindi",
  it: "Italian",
  ja: "Japanese",
  pt: "Brazilian Portuguese",
  zh: "Mandarin Chinese",
};

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const child of children) if (child) node.append(child);
  return node;
}

function pct(v) {
  return v === null || v === undefined ? "-" : `${Math.round(v * 100)}%`;
}

function fmtSeconds(s) {
  s = Math.round(s);
  return s >= 60 ? `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`;
}

// ---------------------------------------------------------------- status & form setup

async function loadStatus() {
  const pill = $("server");
  let status;
  try {
    status = await (await fetch("/api/status")).json();
  } catch (err) {
    pill.className = "server-pill bad";
    pill.textContent = "Studio backend is not reachable";
    return;
  }
  state.status = status;
  $("version").textContent = status.version || "";
  fillPresets(status);
  if (!status.connected) {
    pill.className = "server-pill bad";
    pill.textContent = "Lemonade Server not found. Start it, then reload this page.";
    pill.title = status.error || "";
    $("go").disabled = true;
    return;
  }
  const tk = status.toolkit;
  const omni = tk && tk.omni ? tk.omni : "custom models";
  pill.className = "server-pill ok";
  pill.textContent = `Lemonade ${status.lemonade_version || ""} · ${omni}${status.hardware && status.hardware.length ? " · " + status.hardware[0] : ""}`;
  pill.title = (status.hardware || []).join("\n");
  const select = $("omni");
  for (const m of status.omni || []) {
    const opt = el("option", { value: m.id, text: m.downloaded ? m.id : `${m.id} (not downloaded)` });
    if (!m.downloaded) opt.disabled = true;
    select.append(opt);
  }
  if (!tk) {
    showFormError(status.toolkit_error || "Lemonade has no usable models yet. Try: lemonade pull LMX-Omni-5.5B-Lite");
  }
  $("go").disabled = !tk;
}

function fillPresets(status) {
  const lang = $("language");
  for (const [code, name] of Object.entries(status.languages || {})) {
    lang.append(el("option", { value: code, text: name }));
  }
  lang.value = "en";
  const look = $("look");
  for (const name of status.looks || []) {
    look.append(el("option", { value: name, text: name[0].toUpperCase() + name.slice(1) }));
  }
  fillVoices();
  lang.addEventListener("change", fillVoices);
}

function fillVoices() {
  const status = state.status;
  if (!status) return;
  const code = $("language").value;
  const group = LANGUAGE_VOICE_GROUP[code] || "American English";
  const defaults = (status.default_voices || {})[code] || (status.default_voices || {}).en || ["af_heart", "am_michael"];
  for (const [id, fallback] of [["voice", defaults[0]], ["voice2", defaults[1]]]) {
    const select = $(id);
    select.innerHTML = "";
    const voices = (status.voices || {})[group] || [];
    for (const v of voices) select.append(el("option", { value: v, text: v }));
    select.value = voices.includes(fallback) ? fallback : voices[0] || "";
  }
}

function setupForm() {
  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => {
      state.kind = tab.dataset.kind;
      for (const t of document.querySelectorAll(".tab")) {
        t.classList.toggle("active", t === tab);
        t.setAttribute("aria-selected", t === tab ? "true" : "false");
      }
      for (const pane of document.querySelectorAll(".source")) {
        pane.classList.toggle("hidden", pane.dataset.pane !== state.kind);
      }
    });
  }
  for (const seg of document.querySelectorAll(".seg")) {
    seg.addEventListener("click", (e) => {
      const button = e.target.closest("button");
      if (!button) return;
      for (const b of seg.querySelectorAll("button")) b.classList.toggle("on", b === button);
    });
  }
  const scenes = $("scenes");
  const minutes = $("minutes");
  scenes.addEventListener("input", () => ($("scenes-out").textContent = scenes.value));
  minutes.addEventListener("input", () => ($("minutes-out").textContent = `${minutes.value} min`));

  const drop = $("drop");
  const fileInput = $("file");
  const setFile = (file) => {
    state.file = file;
    $("drop-label").innerHTML = "";
    $("drop-label").append(el("strong", { text: file.name }), el("br"), el("small", { text: `${Math.round(file.size / 1024)} KB · click to change` }));
  };
  fileInput.addEventListener("change", () => fileInput.files[0] && setFile(fileInput.files[0]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("drag");
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  });

  $("record").addEventListener("click", toggleRecording);
  $("go").addEventListener("click", submit);
  $("cancel").addEventListener("click", async () => {
    if (state.jobId) await fetch(`/api/jobs/${state.jobId}/cancel`, { method: "POST" });
  });
}

// ---------------------------------------------------------------- voice memo

async function toggleRecording() {
  const button = $("record");
  if (state.recorder && state.recorder.state === "recording") {
    state.recorder.stop();
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    showFormError("Microphone access was blocked. Allow it in the browser to record a memo.");
    return;
  }
  const type = ["audio/webm", "audio/ogg", "audio/mp4"].find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || "";
  const recorder = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
  const chunks = [];
  const started = Date.now();
  const tick = setInterval(() => {
    const s = Math.floor((Date.now() - started) / 1000);
    $("memo-time").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }, 250);
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = () => {
    clearInterval(tick);
    stream.getTracks().forEach((t) => t.stop());
    const mime = recorder.mimeType || "audio/webm";
    const ext = mime.includes("ogg") ? "ogg" : mime.includes("mp4") ? "m4a" : "webm";
    state.memo = new File(chunks, `memo.${ext}`, { type: mime });
    const audio = $("memo-audio");
    audio.src = URL.createObjectURL(state.memo);
    audio.classList.remove("hidden");
    button.classList.remove("on");
    $("record-label").textContent = "Record again";
  };
  state.recorder = recorder;
  recorder.start();
  button.classList.add("on");
  $("record-label").textContent = "Stop";
}

// ---------------------------------------------------------------- submit

function showFormError(message) {
  const box = $("form-error");
  box.textContent = message;
  box.classList.toggle("hidden", !message);
}

function segValue(id) {
  const on = document.querySelector(`#${id} button.on`);
  return on ? on.dataset.value : null;
}

async function submit() {
  showFormError("");
  const form = new FormData();
  form.append("kind", state.kind);
  if (state.kind === "text") form.append("text", $("text").value);
  if (state.kind === "url") form.append("text", $("url").value);
  if (state.kind === "topic") form.append("text", $("topic").value);
  if (state.kind === "file") {
    if (!state.file) return showFormError("Choose a file first.");
    form.append("file", state.file);
  }
  if (state.kind === "memo") {
    if (!state.memo) return showFormError("Record a memo first.");
    form.append("file", state.memo);
  }
  const options = {
    scenes: Number($("scenes").value),
    minutes: Number($("minutes").value),
    mode: segValue("mode"),
    aspect: segValue("aspect"),
    language: $("language").value,
    look: $("look").value,
    omni: $("omni").value,
    voice: $("voice").value,
    voice2: $("voice2").value,
    audience: $("audience").value,
    critique: $("critique").checked,
    listen_back: $("listen").checked,
    hd: $("hd").checked,
  };
  form.append("options", JSON.stringify(options));
  $("go").disabled = true;
  try {
    const response = await fetch("/api/jobs", { method: "POST", body: form });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Could not start the job");
    startRun(body.id, options);
  } catch (err) {
    showFormError(err.message);
    $("go").disabled = false;
  }
}

// ---------------------------------------------------------------- live board

function startRun(id, options) {
  state.jobId = id;
  state.base = "";
  state.started = Date.now();
  $("empty").classList.add("hidden");
  $("run").classList.remove("hidden");
  $("result").classList.add("hidden");
  $("scenes-grid").innerHTML = "";
  $("toolkit").innerHTML = "";
  $("log").innerHTML = "";
  $("run-title").textContent = "Working…";
  $("cancel").classList.remove("hidden");
  for (const li of document.querySelectorAll("#stages li")) li.className = "";
  clearInterval(state.timer);
  state.timer = setInterval(() => ($("elapsed").textContent = fmtSeconds((Date.now() - state.started) / 1000)), 500);
  state.vertical = options.aspect === "9:16";

  const source = new EventSource(`/api/jobs/${id}/events`);
  state.source = source;
  source.onmessage = (msg) => handle(JSON.parse(msg.data));
  source.addEventListener("end", () => finish(source));
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED) finish(source);
  };
}

function finish(source) {
  source.close();
  clearInterval(state.timer);
  $("go").disabled = false;
  $("cancel").classList.add("hidden");
  loadGallery();
}

function log(message, cls = "") {
  const li = el("li", { text: message });
  if (cls) li.className = cls;
  $("log").append(li);
  $("log").scrollTop = $("log").scrollHeight;
}

function stageItem(stage) {
  return document.querySelector(`#stages li[data-stage="${stage}"]`);
}

function handle(ev) {
  if (ev.base) state.base = ev.base;
  const d = ev.data || {};
  switch (ev.kind) {
    case "stage": {
      const li = stageItem(ev.stage);
      if (li) li.className = d.state === "start" ? "active" : "done";
      if (d.state === "start") log(ev.message);
      break;
    }
    case "log":
      log(ev.message);
      break;
    case "toolkit":
      showToolkit(d.toolkit, d.server);
      break;
    case "storyboard":
      buildScenes(d);
      break;
    case "art":
      showArt(d);
      break;
    case "voice":
      showVoice(d);
      break;
    case "progress":
      if (ev.stage === "edit") {
        // Update one line in place instead of adding a line per segment.
        const last = $("log").lastElementChild;
        if (last && last.dataset.progress === "edit") last.textContent = ev.message;
        else {
          log(ev.message);
          $("log").lastElementChild.dataset.progress = "edit";
        }
      }
      break;
    case "done":
      showResult(d);
      break;
    case "error": {
      $("run-title").textContent = ev.message === "Cancelled" ? "Cancelled" : "Something went wrong";
      log(ev.message, "err");
      const active = document.querySelector("#stages li.active");
      if (active) active.className = "failed";
      break;
    }
  }
}

function chip(label, value) {
  return el("span", { class: "chip" }, `${label} `, el("b", { text: value }));
}

function showToolkit(tk, server) {
  const box = $("toolkit");
  box.innerHTML = "";
  if (tk.omni) box.append(chip("Omni", tk.omni));
  const roles = tk.roles || {};
  for (const [role, label] of [["writer", "Writer"], ["illustrator", "Illustrator"], ["voice", "Voice"], ["listener", "Listener"]]) {
    if (roles[role]) box.append(chip(label, roles[role].id));
  }
  if (server && server.processor) box.append(chip("on", server.processor));
}

function buildScenes(d) {
  $("run-title").textContent = d.title;
  state.hosts = d.hosts || ["A", "B"];
  const grid = $("scenes-grid");
  grid.innerHTML = "";
  d.scenes.forEach((scene, i) => {
    const pic = el("div", { class: "pic" }, el("span", { class: "num", text: String(i + 1).padStart(2, "0") }));
    if (state.vertical) pic.style.aspectRatio = "9 / 16";
    const text = scene.lines && scene.lines.length
      ? scene.lines.map((l) => `${l.speaker === "B" ? state.hosts[1] : state.hosts[0]}: ${l.text}`).join("\n")
      : scene.narration;
    const narration = el("p", { text });
    narration.style.whiteSpace = "pre-line";
    const card = el(
      "article",
      { class: "scene", id: `scene-${i}` },
      pic,
      el("div", { class: "body" }, el("h3", { text: scene.caption }), narration, el("div", { class: "badges" }))
    );
    grid.append(card);
  });
}

function badge(text, cls = "") {
  return el("span", { class: `badge ${cls}`.trim(), text });
}

function showArt(d) {
  const card = $(`scene-${d.index}`);
  if (!card) return;
  const pic = card.querySelector(".pic");
  const img = el("img", { src: `${state.base}${d.image}?v=${Date.now()}`, alt: "" });
  pic.querySelectorAll("img").forEach((n) => n.remove());
  pic.prepend(img);
  pic.classList.add("ready");
  const badges = card.querySelector(".badges");
  if (d.fallback) badges.append(badge("Text slide", "warn"));
  else if (d.score) {
    const cls = d.score >= 4 ? "good" : d.score <= 2 ? "warn" : "";
    badges.append(badge(`Critic ${d.score}/5${d.attempts > 1 ? ` · redrawn ${d.attempts - 1}×` : ""}`, cls));
  }
}

function showVoice(d) {
  const card = $(`scene-${d.index}`);
  if (!card) return;
  const body = card.querySelector(".body");
  const audio = el("audio", { controls: "", preload: "metadata", src: `${state.base}${d.audio}` });
  body.insertBefore(audio, card.querySelector(".badges"));
  const badges = card.querySelector(".badges");
  if (d.wer !== null && d.wer !== undefined) {
    const changed = d.first_wer !== null && d.first_wer !== undefined && d.first_wer !== d.wer;
    const cls = d.wer <= 0.1 ? "good" : d.wer > 0.25 ? "warn" : "";
    const b = badge(`Heard back: ${changed ? pct(d.first_wer) + " → " : ""}${pct(d.wer)} error`, cls);
    if (d.heard) b.title = `Listener heard: ${d.heard}`;
    badges.append(b);
  }
  for (const [word, spoken] of Object.entries(d.fixes || {})) {
    badges.append(badge(`${word} → “${spoken}”`, "fix"));
  }
}

function showResult(d) {
  $("run-title").textContent = $("run-title").textContent === "Working…" ? "Done" : $("run-title").textContent;
  for (const li of document.querySelectorAll("#stages li")) if (li.className !== "failed") li.className = "done";
  if (!d.video) {
    log(`Storyboard saved in ${d.workdir}`);
    return;
  }
  $("result").classList.remove("hidden");
  const video = $("video");
  video.src = `${state.base}${d.video}`;
  video.poster = `${state.base}thumbnail.jpg`;
  $("dl-video").href = `${state.base}${d.video}`;
  $("dl-srt").href = `${state.base}captions.srt`;
  $("open-report").href = `${state.base}${d.report}`;
  $("result-path").textContent = `${fmtSeconds(d.duration)} video made in ${fmtSeconds(d.seconds)} · saved in ${d.workdir}`;
  log(`Finished: ${fmtSeconds(d.duration)} of video in ${fmtSeconds(d.seconds)}`);
}

// ---------------------------------------------------------------- gallery

async function loadGallery() {
  let reels = [];
  try {
    reels = await (await fetch("/api/reels")).json();
  } catch (err) {
    return;
  }
  const box = $("gallery");
  if (!reels.length) return;
  box.innerHTML = "";
  for (const r of reels) {
    const card = el(
      "a",
      { class: "reel-card", href: `${r.base}report.html`, target: "_blank", rel: "noopener" },
      el("img", { src: `${r.base}thumbnail.jpg`, alt: "", loading: "lazy" }),
      el("div", {}, el("strong", { text: r.title || "Untitled" }), el("small", { text: `${fmtSeconds(r.duration || 0)} · made in ${fmtSeconds(r.made_in || 0)}${r.omni ? " · " + r.omni : ""}` }))
    );
    box.append(card);
  }
}

setupForm();
loadStatus();
loadGallery();
