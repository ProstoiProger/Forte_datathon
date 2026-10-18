"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  drop: $("drop"), file: $("file"), fileName: $("file-name"), run: $("run"),
  status: $("status"), fill: $("fill"), statusText: $("status-text"), error: $("error"),
  result: $("result"), verdict: $("verdict"), video: $("video"), meta: $("meta"),
  events: $("events"), types: $("types"), lightbox: $("lightbox"), lightboxImg: $("lightbox-img"),
  gate: $("gate"), gateForm: $("gate-form"), keyInput: $("key-input"), keyToggle: $("key-toggle"),
  keyError: $("key-error"), keySubmit: $("key-submit"), keyChange: $("key-change"),
};

let selected = null;
let objectUrl = null;

// The service needs the same X-API-Key as the API. The key is kept in this browser only.
const KEY_STORE = "cashwatch.apiKey";

class AuthError extends Error {}

function loadKey() {
  try { return localStorage.getItem(KEY_STORE) || ""; } catch (_) { return ""; }
}

function saveKey(key) {
  try {
    if (key) localStorage.setItem(KEY_STORE, key);
    else localStorage.removeItem(KEY_STORE);
  } catch (_) { /* storage blocked: the key lives until the page is closed */ }
}

let apiKey = loadKey();

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
document.addEventListener("keydown", (e) => { if (e.key === "Escape") els.lightbox.hidden = true; });

function upload(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.open("POST", "/ui/api/analyze");
    xhr.setRequestHeader("X-API-Key", apiKey);
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) setProgress("Загрузка", (e.loaded / e.total) * 100);
    });
    xhr.addEventListener("load", () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (_) { body = {}; }
      if (xhr.status === 200 && body.job_id) resolve(body.job_id);
      else if (xhr.status === 401) reject(new AuthError("Ключ доступа не принят"));
      else reject(new Error(body.detail || `Ошибка загрузки (${xhr.status})`));
    });
    xhr.addEventListener("error", () => reject(new Error("Сервис недоступен")));
    xhr.send(form);
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitForJob(jobId) {
  for (;;) {
    const response = await fetch(`/ui/api/jobs/${jobId}`, { headers: { "X-API-Key": apiKey } });
    if (response.status === 401) throw new AuthError("Ключ доступа не принят");
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
  if (!rule) return el("div", { class: "rule rule-none", text: "Нормативное основание: не указано" });
  const unverified = rule.verified === false;
  const label = unverified
    ? `Возможное основание (подобрано поиском по НПА, требует проверки): ${rule.clause}. `
    : `Основание: ${rule.approved_by}, ${rule.clause}. `;
  const tag = unverified
    ? el("span", { class: "tag tag-outline tag-warn", text: "требует проверки" })
    : el("span", { class: "tag tag-solid", text: "норма из реестра" });
  const text = [el("span", { text: label })];
  if (rule.url) text.push(el("a", { href: rule.url, target: "_blank", rel: "noopener", text: "Текст акта" }));
  const parts = [el("div", { class: "rule-head" }, [tag]), el("p", { class: "rule-text" }, text)];
  if (rule.quote) parts.push(el("blockquote", { class: "rule-quote", text: `«${rule.quote}»` }));
  return el("div", { class: `rule ${unverified ? "rule-unverified" : "rule-verified"}` }, parts);
}

function eventCard(v, index) {
  const frames = v.evidence_frames.map((f) => {
    const src = `data:image/jpeg;base64,${f.image_base64}`;
    const img = el("img", { src, alt: `Кадр ${fmtTime(f.timestamp_sec)}` });
    const thumb = el("button", { class: "thumb", type: "button", "aria-label": `Увеличить кадр ${fmtTime(f.timestamp_sec)}`, onclick: () => {
      els.lightboxImg.src = src;
      els.lightbox.hidden = false;
    } }, [img]);
    return el("figure", { class: "frame" }, [thumb, el("figcaption", { text: fmtTime(f.timestamp_sec) })]);
  });
  const conf = typeof v.confidence === "number" ? el("span", { class: "conf", text: `уверенность ${(v.confidence * 100).toFixed(0)}%` }) : null;
  return el("article", { class: "card event" }, [
    el("h3", { class: "event-title" }, [
      el("span", { class: "event-num", text: String(index + 1) }),
      el("span", { text: v.type }),
    ]),
    el("div", { class: "event-head" }, [
      el("button", { class: "tc", type: "button", "aria-label": `Перейти к ${fmtTime(v.timecode_sec)} в видео`, onclick: () => seek(v.timecode_sec), text: fmtTime(v.timecode_sec) }),
      conf,
    ]),
    el("p", { class: "event-desc", text: v.description }),
    frames.length ? el("div", { class: "frames" }, frames) : null,
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
    : [el("p", { class: "card empty-state", text: "Визуально наблюдаемых признаков нарушений не найдено." })]));
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
    els.status.hidden = true;
    if (err instanceof AuthError) lock("Ключ доступа не принят. Введите действующий ключ и запустите проверку снова.");
    else showError(err.message);
  } finally {
    els.run.disabled = !selected;
  }
});

function renderTypes(types) {
  els.types.replaceChildren(...types.map((t) => el("li", { class: "type-item" }, [
    el("strong", { class: "type-title", text: t.title }),
    el("span", { class: "type-desc", text: t.rationale }),
  ])));
}

// Checks a key against the service; the list of violation types is the cheapest call that needs one.
async function fetchTypes(key) {
  let response;
  try {
    response = await fetch("/ui/api/types", { headers: { "X-API-Key": key } });
  } catch (_) {
    throw new Error("Сервис недоступен, попробуйте позже");
  }
  if (response.status === 401) throw new AuthError("Неверный ключ доступа");
  if (!response.ok) throw new Error(`Сервис недоступен (${response.status})`);
  return response.json();
}

function showKeyError(message) {
  els.keyError.textContent = message;
  els.keyError.hidden = !message;
}

function lock(message = "") {
  apiKey = "";
  saveKey("");
  els.keyChange.hidden = true;
  els.keyInput.value = "";
  showKeyError(message);
  els.gate.hidden = false;
  document.body.classList.add("locked");
  els.keyInput.focus();
}

function unlock(key, types) {
  apiKey = key;
  saveKey(key);
  els.gate.hidden = true;
  els.keyChange.hidden = false;
  document.body.classList.remove("locked");
  showKeyError("");
  renderTypes(types);
}

els.gateForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const key = els.keyInput.value.trim();
  if (!key) {
    showKeyError("Введите ключ доступа");
    els.keyInput.focus();
    return;
  }
  els.keySubmit.disabled = true;
  showKeyError("");
  try {
    unlock(key, await fetchTypes(key));
  } catch (err) {
    showKeyError(err.message);
    els.keyInput.select();
  } finally {
    els.keySubmit.disabled = false;
  }
});

els.keyToggle.addEventListener("click", () => {
  const show = els.keyInput.type === "password";
  els.keyInput.type = show ? "text" : "password";
  els.keyToggle.textContent = show ? "Скрыть" : "Показать";
  els.keyToggle.setAttribute("aria-pressed", String(show));
});

els.keyChange.addEventListener("click", () => lock());

if (!apiKey) {
  lock();
} else {
  fetchTypes(apiKey)
    .then((types) => unlock(apiKey, types))
    .catch((err) => {
      if (err instanceof AuthError) lock("Сохранённый ключ больше не действует. Введите новый ключ.");
      else els.keyChange.hidden = false; // service down: keep the key, the upload will report the error
    });
}
