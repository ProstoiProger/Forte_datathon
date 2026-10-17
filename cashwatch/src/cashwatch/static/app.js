"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  drop: $("drop"), file: $("file"), fileName: $("file-name"), run: $("run"),
  status: $("status"), fill: $("fill"), statusText: $("status-text"), error: $("error"),
  result: $("result"), verdict: $("verdict"), video: $("video"), meta: $("meta"),
  events: $("events"), types: $("types"), lightbox: $("lightbox"), lightboxImg: $("lightbox-img"),
};

let selected = null;
let objectUrl = null;

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") node.textContent = v;
    else if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children) if (child) node.append(child);
  return node;
}

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = (sec - m * 60).toFixed(1).padStart(4, "0");
  return `${String(m).padStart(2, "0")}:${s}`;
}

function fmtSize(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
}

function setProgress(label, pct) {
  els.status.hidden = false;
  els.fill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  els.statusText.textContent = `${label} ${Math.round(pct)}%`;
}

function showError(message) {
  els.error.textContent = message;
  els.error.hidden = false;
  els.run.disabled = !selected;
}

function choose(file) {
  if (!file) return;
  selected = file;
  els.fileName.textContent = `${file.name} (${fmtSize(file.size)})`;
  els.run.disabled = false;
  els.error.hidden = true;
}

els.file.addEventListener("change", () => choose(els.file.files[0]));
["dragenter", "dragover"].forEach((ev) => els.drop.addEventListener(ev, (e) => {
  e.preventDefault();
  els.drop.classList.add("over");
}));
["dragleave", "drop"].forEach((ev) => els.drop.addEventListener(ev, (e) => {
  e.preventDefault();
  els.drop.classList.remove("over");
}));
els.drop.addEventListener("drop", (e) => choose(e.dataTransfer.files[0]));
els.lightbox.addEventListener("click", () => { els.lightbox.hidden = true; });

function upload(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.open("POST", "/ui/api/analyze");
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) setProgress("Загрузка", (e.loaded / e.total) * 100);
    });
    xhr.addEventListener("load", () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (_) { body = {}; }
      if (xhr.status === 200 && body.job_id) resolve(body.job_id);
      else reject(new Error(body.detail || `Ошибка загрузки (${xhr.status})`));
    });
    xhr.addEventListener("error", () => reject(new Error("Сервис недоступен")));
    xhr.send(form);
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitForJob(jobId) {
  for (;;) {
    const response = await fetch(`/ui/api/jobs/${jobId}`);
    if (!response.ok) throw new Error(`Ошибка запроса статуса (${response.status})`);
    const body = await response.json();
    if (body.status === "done") return body;
    if (body.status === "failed") throw new Error(`Обработка не удалась: ${body.error || "неизвестная ошибка"}`);
    setProgress("Анализ", (body.progress || 0) * 100);
    await sleep(1000);
  }
}

function seek(sec) {
  els.video.currentTime = sec;
  els.video.pause();
  els.video.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function ruleBlock(rule) {
  if (!rule) return el("div", { class: "rule", text: "Нормативное основание: не указано" });
  const parts = [el("span", { text: `Основание: ${rule.approved_by}, ${rule.clause}. ` })];
  if (rule.url) parts.push(el("a", { href: rule.url, target: "_blank", rel: "noopener", text: "Текст акта" }));
  if (rule.quote) parts.push(el("p", { text: `"${rule.quote}"` }));
  return el("div", { class: "rule" }, parts);
}

function eventCard(v, index) {
  const frames = v.evidence_frames.map((f) => {
    const src = `data:image/jpeg;base64,${f.image_base64}`;
    const img = el("img", { src, alt: `Кадр ${fmtTime(f.timestamp_sec)}`, onclick: () => {
      els.lightboxImg.src = src;
      els.lightbox.hidden = false;
    } });
    return el("figure", {}, [img, el("figcaption", { text: fmtTime(f.timestamp_sec) })]);
  });
  const conf = typeof v.confidence === "number" ? el("span", { class: "conf", text: `уверенность ${(v.confidence * 100).toFixed(0)}%` }) : null;
  return el("article", { class: "card event" }, [
    el("h3", { text: `${index + 1}. ${v.type}` }),
    el("div", { class: "event-head" }, [
      el("button", { class: "tc", onclick: () => seek(v.timecode_sec), text: fmtTime(v.timecode_sec) }),
      conf,
    ]),
    el("p", { text: v.description }),
    el("div", { class: "frames" }, frames),
    ruleBlock(v.rule),
  ]);
}

function render(result) {
  els.result.hidden = false;
  const bad = result.verdict === "violation";
  els.verdict.className = `verdict ${bad ? "bad" : "ok"}`;
  els.verdict.textContent = bad
    ? `Обнаружены признаки возможных нарушений: ${result.violations.length}`
    : "Нарушение не обнаружено";
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(selected);
  els.video.src = objectUrl;
  const parts = [];
  if (result.duration_sec) parts.push(`длительность ${fmtTime(result.duration_sec)}`);
  if (result.processing_sec) parts.push(`обработка ${result.processing_sec.toFixed(1)} с`);
  els.meta.textContent = parts.join(", ");
  els.events.replaceChildren(...(bad
    ? result.violations.map(eventCard)
    : [el("p", { class: "card", text: "Визуально наблюдаемых признаков нарушений не найдено." })]));
  els.result.scrollIntoView({ behavior: "smooth" });
}

els.run.addEventListener("click", async () => {
  if (!selected) return;
  els.run.disabled = true;
  els.error.hidden = true;
  els.result.hidden = true;
  try {
    setProgress("Загрузка", 0);
    const jobId = await upload(selected);
    setProgress("Анализ", 0);
    const result = await waitForJob(jobId);
    setProgress("Готово", 100);
    render(result);
  } catch (err) {
    showError(err.message);
  } finally {
    els.run.disabled = !selected;
  }
});

fetch("/ui/api/types")
  .then((r) => (r.ok ? r.json() : []))
  .then((types) => {
    els.types.replaceChildren(...types.map((t) => el("li", {}, [
      el("strong", { text: t.title }),
      el("span", { text: ` - ${t.rationale}` }),
    ])));
  })
  .catch(() => {});
