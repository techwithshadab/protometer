// Aegis console — light executive report over the Protected-Pipeline API.
// The frontend is served by the same FastAPI server, so the API base is the page origin.
const API = (window.AMLGUARD_API_BASE || window.location.origin) + "/api";
const $ = s => document.querySelector(s);
// Escape ALL HTML-significant characters before any string reaches innerHTML. User input and
// server text (guardrail explanations, entity types, model replies) are untrusted.
const esc = s => String(s ?? "").replace(/[&<>"']/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
const get = async p => (await fetch(API + p)).json();

// Title Case for stage names and headings, so every title reads as a proper heading (PascalCase-ish
// Title Case). Small connector words stay lowercase unless first.
const SMALL = new Set(["a","an","and","as","at","but","by","for","in","of","on","or","the","to","vs","via","over"]);
const titleCase = s => String(s ?? "").split(/\s+/).map((w, i) => {
  if (w.includes("-")) return w.split("-").map(p => cap(p)).join("-");
  const lw = w.toLowerCase();
  return (i > 0 && SMALL.has(lw)) ? lw : cap(w);
}).join(" ");
const cap = w => w ? w[0].toUpperCase() + w.slice(1) : w;

// Format a number for display at a consistent precision. Integers stay integers; fractions round to
// 3 decimals (so a stray 0.3389255804451789 reads as 0.339). Non-numbers pass through unchanged.
const fmtNum = (v, dp = 3) => {
  const n = Number(v);
  if (v === null || v === undefined || v === "" || Number.isNaN(n)) return String(v ?? "");
  return Number.isInteger(n) ? String(n) : n.toFixed(dp);
};

const done = new Set();
let stages = [], mode = "replay", currentDomain = "aml";

// Per-domain lede copy (Title Case headline + one sentence). Keeps the executive framing focused
// on the question each domain answers.
// Per-view (pipeline | assistant) × per-domain lede, so the headline always describes what the
// current view actually shows (the pipeline's stage-by-stage measurement, or the assistant's
// protected conversation) rather than a generic pipeline blurb.
const LEDE = {
  pipeline: {
    aml: ["Run A Real Investigation On Protected Data",
      "A classifier ranks the alert queue and the model writes case notes, entirely over tokenized identities, re-identified only for the role entitled to see them. Below is what that protection costs each stage, measured on a live run."],
    healthcare: ["Release Patient Data For AI Without Breaking HIPAA",
      "Two de-identification standards, measured: Safe Harbor removes the direct identifiers; Expert Determination quantifies the residual re-identification risk before and after k-anonymization. Reported honestly, including where the bar is not met."],
    "customer-support": ["Least-Privilege Access At The Point Of Contact",
      "The same protected reply, re-identified two ways: a front-line agent sees the customer masked, a supervisor may reveal in full. One tokenized message, role-gated at the presentation boundary."],
  },
  assistant: {
    aml: ["An Investigation Assistant That Never Sees A Name",
      "Ask about a case; the assistant reasons entirely over tokens, every reply is scanned before you see it, and identities are revealed only for the role entitled to them. Step through a recorded conversation, or run one live."],
    healthcare: ["A Clinical Assistant On De-Identified Records",
      "Ask about a patient; the assistant works over tokenized records, its replies are scanned at the boundary, and who sees the real identifiers depends on the role. Step through a recorded conversation, or run one live."],
    "customer-support": ["A Support Assistant That Protects The Customer",
      "Handle a case without exposing the customer: the assistant reasons over tokens, replies are scanned before release, and full details unlock only for an entitled role. Step through a recorded conversation, or run one live."],
  },
};

async function boot() {
  const domains = await get("/domains");
  $("#domain").innerHTML = domains.map(d => `<option value="${esc(d.name)}">${esc(titleCase(d.label))}</option>`).join("");
  currentDomain = $("#domain").value;
  await loadPipeline(currentDomain);
  $("#replay").onclick = () => setMode("replay");
  $("#live").onclick = () => setMode("live");
  $("#usecase").onchange = switchView;
  $("#domain").onchange = async () => {
    currentDomain = $("#domain").value;
    setLede(currentDomain);
    await loadPipeline(currentDomain);
    renderLiveNote();   // the live-availability note is domain-specific
    if ($("#usecase").value === "chatbot") renderChat();
  };
  setLede(currentDomain);
  // Sync the Live button's "$" cost badge to the resolved model at load, not just in Live mode: a
  // local ($0) model should never show a cost marker, even while the user is still in Replay.
  loadLiveHealth().then(h => {
    const costBadge = $("#live .cost");
    if (costBadge) costBadge.style.display = (h.provider === "ollama") ? "none" : "";
  });
}

// Provenance bar: the "View" reflects the actual Batch Analysis / Live Assistant toggle (not a
// hardcoded "Use Case Batch"), so it stays truthful when you switch views.
let provMeta = null;
function renderProvenance() {
  if (!provMeta) return;
  const view = ($("#usecase") && $("#usecase").value === "chatbot") ? "Live Assistant" : "Batch Analysis";
  const dot = `<span class="dot"></span>`;
  $("#prov").innerHTML = [
    `<span class="kv">Domain <b>${esc(titleCase(provMeta.domain))}</b></span>`,
    `<span class="kv">View <b>${esc(view)}</b></span>`,
    `<span class="kv">Corpus <b>${esc(provMeta.corpus)}</b></span>`,
    `<span class="kv">Model <b>${esc(provMeta.model)}</b></span>`,
  ].join(dot);
}

function setLede(domain) {
  const view = ($("#usecase") && $("#usecase").value === "chatbot") ? "assistant" : "pipeline";
  const group = LEDE[view] || LEDE.pipeline;
  const [h, p] = group[domain] || group.aml;
  $("#lede-h").textContent = h;
  $("#lede-p").textContent = p;
}

async function loadPipeline(domain) {
  const pipe = await get("/pipeline?domain=" + encodeURIComponent(domain));
  stages = pipe.stages;
  done.clear();
  provMeta = { domain: pipe.domain, corpus: pipe.corpus_fingerprint, model: pipe.model };
  renderProvenance();
  await renderJourneyContext(domain);
  renderFlow();
  $("#panel").innerHTML = `<div class="placeholder">Select a stage above to follow one record through it: the data transformation and what protection costs, together.</div>`;
}

// A LEAN context strip above the stepper: names the one synthetic record every stage traces, and a
// legend for the clear-vs-token colour code. The full per-stage transformation now lives INSIDE each
// stage's panel (one unified place), so there is no separate journey rail duplicating it.
async function renderJourneyContext(domain) {
  const host = $("#journey");
  if (!host) return;
  const j = await getJourney(domain);
  if (!j || j.detail) { host.innerHTML = ""; return; }
  host.innerHTML = `
    <div class="jcontext">
      <div class="jctext"><b>Following one record:</b> ${esc(j.subject_name_clear)} <span class="syn">(synthetic)</span>,
        with ${esc(String(j.identity_tokens))} identity spans tokenized. Click any stage to see what happens to it.</div>
      <div class="jlegend">
        <span class="lg"><span class="sw clear"></span>cleartext identifier</span>
        <span class="lg"><span class="sw tok"></span>protection token</span>
        <span class="lg"><span class="sw mask"></span>masked for role</span>
      </div>
    </div>`;
}

// Highlight cleartext PII (red) in the ONE clear sample, and protection tokens (accent) elsewhere.
function markClearPii(text) {
  let s = esc(text);
  // Names/SSN/email/phone-shaped substrings in the clear sample -> red chips (illustrative).
  s = s.replace(/\b\d{3}-\d{2}-\d{4}\b/g, m => `<span class="clearpii">${m}</span>`);
  s = s.replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g, m => `<span class="clearpii">${m}</span>`);
  s = s.replace(/\b\d{3}-\d{3}-\d{4}\b/g, m => `<span class="clearpii">${m}</span>`);
  return s;
}
function markTokens(escaped) {
  // A committed reply can carry a token whose inner value was PII-SHAPE-redacted before persisting
  // (e.g. the model wrote an SSN-shaped token value, redacted to `[SSN]`), leaving the nested form
  // `[SOCIAL_SECURITY_ID][SSN][/SOCIAL_SECURITY_ID]`. That reads as broken markup, so collapse any
  // `[TYPE][SHAPE][/TYPE]` (SHAPE ∈ SSN/CARD/EMAIL/PHONE, the redaction placeholders) into a single
  // clean masked chip — which is exactly what it is: a protected identifier with its value withheld.
  let s = escaped.replace(/\[([A-Z_]+)\]\[(SSN|CARD|EMAIL|PHONE)\]\[\/\1\]/g,
    (_m, type) => `<span class="masked-chip">[${type}: masked]</span>`);
  // wrap the remaining well-formed [TAG]…[/TAG] token spans
  return s.replace(/\[([A-Z_]+)\][^\[]*?\[\/\1\]/g, m => `<span class="tag">${m}</span>`);
}

// Normalize em dashes out of model-authored reply prose at render time, so the UI never shows one.
// The committed transcript stays a faithful record of what the model wrote; only the presentation is
// normalized. A spaced em dash after a heading/label (bold close, or a "Title Case:" style lead-in)
// reads as a colon; every other spaced em dash is an aside, which reads as a comma pause. An unspaced
// em dash between words is also an aside -> comma.
function deEmDash(text) {
  return String(text ?? "")
    // "**Label** — elaboration"  and  "# Heading — subject"  -> colon
    .replace(/(\*\*|#{1,6} .*?) — /g, "$1: ")
    // any remaining spaced em dash -> comma pause
    .replace(/ — /g, ", ")
    // unspaced em dash between words -> comma
    .replace(/(\S)—(\S)/g, "$1, $2");
}

// Render the small Markdown subset the model emits (headings, bold, bullets, horizontal rules,
// paragraphs) into HTML, so replies read as formatted text instead of showing raw `**`/`##`/`---`.
// Input is ALREADY html-escaped (esc ran first), so no tag in the source can be user markup; this
// only introduces our own <strong>/<hr>/<ul> wrappers. Runs after markTokens/deEmDash.
function mdLite(escaped) {
  const lines = String(escaped ?? "").split("\n");
  const html = [];
  let inList = false;
  const closeList = () => { if (inList) { html.push("</ul>"); inList = false; } };
  for (let raw of lines) {
    const line = raw.trimEnd();
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { closeList(); html.push("<hr>"); continue; }
    if (/^\s*$/.test(line)) { closeList(); continue; }
    let h = line.match(/^\s*(#{1,6})\s+(.*)$/);
    if (h) { closeList(); html.push(`<div class="md-h">${inline(h[2])}</div>`); continue; }
    let b = line.match(/^\s*[-*+]\s+(.*)$/);
    if (b) { if (!inList) { html.push("<ul>"); inList = true; } html.push(`<li>${inline(b[1])}</li>`); continue; }
    closeList(); html.push(`<div class="md-p">${inline(line)}</div>`);
  }
  closeList();
  return html.join("");
  // Inline spans: **bold** and `code`. (Tokens/masked chips were already wrapped upstream.)
  function inline(s) {
    return s
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }
}

function setMode(m) {
  mode = m;
  $("#replay").classList.toggle("on", m == "replay");
  $("#live").classList.toggle("on", m == "live");
  renderLiveNote();
  if ($("#usecase").value === "chatbot") renderChat();
}

// Which model answers a LIVE turn depends on the deployment: a hosted model when cloud credentials
// are present, otherwise the open-source local model. Surfacing it (only in Live mode) tells a user
// who forked the repo without credentials that live turns will run locally and free — or what to do
// if the local model is not set up yet. Fetched once and cached.
let liveHealth = null;      // the /health .live_chat block (model + readiness)
let liveDomains = null;     // which domains the server reports as live-capable
async function loadLiveHealth() {
  if (liveHealth) return liveHealth;
  try {
    const h = await get("/health");
    liveHealth = h.live_chat || {};
    liveDomains = h.live_domains || [];
  } catch { liveHealth = {}; liveDomains = []; }
  return liveHealth;
}
// A domain supports a live turn only if the server lists it. Until /health has loaded we don't
// know, so treat it as live-capable (optimistic) — the note/gate correct themselves once loaded.
function domainSupportsLive(d) { return liveDomains === null || liveDomains.includes(d); }
async function renderLiveNote() {
  const n = $("#livenote"); if (!n) return;
  if (mode !== "live") { n.hidden = true; return; }
  const h = await loadLiveHealth();
  // Domain gate first: if the server reports this domain as not live-capable, say so plainly
  // and point at Replay/Batch, rather than letting the turn 503 with a raw error.
  if (!domainSupportsLive(currentDomain)) {
    n.hidden = false;
    n.className = "livenote warn";
    n.innerHTML = `Live chat is not enabled for this domain. It demonstrates the same protection through <b>Replay</b> (a recorded conversation) and the <b>Batch Analysis</b> stepper.`;
    return;
  }
  // The "$" badge on the Live button means "a real billed call"; a local model is free, so hide it.
  const costBadge = $("#live .cost");
  if (costBadge) costBadge.style.display = (h.provider === "ollama") ? "none" : "";
  n.hidden = false;
  const model = esc(h.model || "the configured model");
  if (h.provider === "ollama") {
    if (h.ready) {
      n.className = "livenote ok";
      n.innerHTML = `Live turns run on the local open-source model <b>${model}</b> (no cloud account, $0 per turn). Tokenization still uses Protegrity Developer Edition.`;
    } else if (h.ollama_reachable === false) {
      n.className = "livenote warn";
      n.innerHTML = `Live turns need a model. No cloud credentials were found, and <b>Ollama</b> is not running. Install it from <code>ollama.com</code> and run <code>make setup-local-model</code>, then reload. (Replay mode works without any of this.)`;
    } else {
      n.className = "livenote warn";
      n.innerHTML = `The local model <b>${model}</b> is not downloaded yet. Run <code>make setup-local-model</code> once (or set <code>AMLGUARD_AUTO_PULL_MODEL=true</code>), then reload. (Replay mode works without it.)`;
    }
  } else if (h.model) {
    n.className = "livenote ok";
    n.innerHTML = `Live turns run on the hosted model <b>${model}</b> (a real billed call).`;
  } else {
    n.className = "livenote warn";
    n.innerHTML = `Live chat is not configured on this deployment. Replay mode replays a recorded run.`;
  }
}

function switchView() {
  const cb = $("#usecase").value === "chatbot";
  $("#batch").style.display = cb ? "none" : "block";
  $("#chatbot").style.display = cb ? "block" : "none";
  setLede(currentDomain);       // headline reflects the view (pipeline vs assistant)
  renderProvenance();           // provenance "View" reflects the toggle
  if (cb) renderChat();
}

function renderFlow() {
  const f = $("#flow"); f.innerHTML = "";
  stages.forEach((s, i) => {
    const n = el("div", "node" + (done.has(s.id) ? " done" : ""));
    n.innerHTML = `<div class="num">${done.has(s.id) ? "✓" : i + 1}</div>
      <div class="label"><span class="t">${esc(titleCase(s.title))}</span>
      <span class="s">${esc(s.subtitle)}</span></div>`;
    n.onclick = () => runStage(s, n);
    f.appendChild(n);
  });
}

async function runStage(s, node) {
  document.querySelectorAll(".node").forEach(x => x.classList.remove("active"));
  node.classList.add("active");
  const p = $("#panel");
  if (mode === "live") return runStageLive(s, node, p);
  // Header first, then a subtle shimmer that shows ONLY while the committed artifact actually
  // fetches (real latency) — no scripted fixed-duration bar. revealStage does the fetch; when it
  // returns we drop the shimmer and fade the content in.
  p.innerHTML = `<div class="eyebrow">${esc(titleCase(s.subtitle))}</div>
    <h2>${esc(titleCase(s.title))}</h2><div class="measures">${esc(s.measures)}</div>
    <div class="loading-shimmer" aria-label="loading"></div>`;
  await revealStage(s, p);
  const sh = p.querySelector(".loading-shimmer"); if (sh) sh.remove();
  done.add(s.id); node.classList.add("done");
  addNext(s, p);
}

function addNext(s, p) {
  const idx = stages.indexOf(s);
  if (idx < stages.length - 1) {
    const nx = el("button", "btn", "Next · " + titleCase(stages[idx + 1].title));
    nx.style.marginTop = "20px";
    nx.onclick = () => document.querySelectorAll(".node")[idx + 1].click();
    p.appendChild(nx);
  }
}

// ── metric / plot helpers ───────────────────────────────────────────────────
const metric = (v, l) => `<div class="metric"><div class="v">${v}</div><div class="l">${l}</div></div>`;
const metrics = arr => `<div class="metrics">${arr.join("")}</div>`;
const plot = (scope, name, cap) =>
  `<div class="plot"><div class="cap">${esc(cap)}</div><img src="${API}/plot/${esc(scope)}/${esc(name)}" alt="${esc(cap)}"
     onerror="this.parentNode.className='plot err';this.parentNode.innerHTML='${esc(cap)}: not available (train run required)'"/></div>`;

// ── Live batch ──────────────────────────────────────────────────────────────
async function runStageLive(s, node, p, confirmToken) {
  p.innerHTML = `<div class="eyebrow">${esc(titleCase(s.subtitle))}</div>
    <h2>${esc(titleCase(s.title))}</h2><div class="measures">${esc(s.measures)}</div>
    <div class="running">Running Live</div>`;
  let r;
  try {
    r = await (await fetch(API + "/batch/run-stage", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ domain: currentDomain, stage: s.id, scope: "none", confirm_token: confirmToken || null })
    })).json();
  } catch (e) { p.innerHTML = `<h2>${esc(titleCase(s.title))}</h2><div class="note">Error: ${esc(String(e))}</div>`; return; }

  if (r.status === "confirm_required") {
    p.querySelector(".running").remove();
    const box = el("div", "confirm", `<div class="cost">Live Run · $${esc(String(r.estimate.cost_usd))}${r.estimate.calls ? ` · ~${esc(String(r.estimate.calls))} LLM calls` : ""}</div>
      <div class="footnote">A real hosted-model call. Its results are kept separate and never overwrite the recorded run.</div>
      <button class="btn" id="confirmRun" style="margin-top:14px">Confirm And Run · $${esc(String(r.estimate.cost_usd))}</button>`);
    p.appendChild(box);
    $("#confirmRun").onclick = () => runStageLive(s, node, p, r.confirm_token);
    return;
  }
  if (r.status !== "done") {
    p.innerHTML = `<h2>${esc(titleCase(s.title))}</h2><div class="note">${esc(r.detail || r.message || "Could not run")}</div>`;
    return;
  }
  p.innerHTML = `<div class="eyebrow">${esc(titleCase(s.subtitle))}</div>
    <h2>${esc(titleCase(s.title))} <span class="live-badge">LIVE · ${esc((r.run_id || "").slice(0, 8))}</span></h2>
    <div class="measures">${esc(s.measures)}</div>`;
  revealLiveArtifact(s, r.artifact, p);
  done.add(s.id); node.classList.add("done");
  addNext(s, p);
}

function revealLiveArtifact(s, artifact, p) {
  const body = el("div");
  if (!artifact) body.innerHTML = `<div class="note">Ran live (local step); no measured result to show.</div>`;
  else if (s.id === "train")
    body.innerHTML = metrics([
      metric(esc((artifact.average_precision || 0).toFixed(3)), "Average Precision (this live run)"),
      metric(esc((artifact.roc_auc || 0).toFixed(3)), "ROC-AUC"),
    ]) + `<div class="note">Trained live on the protected ledger; written to the isolated live run.</div>`;
  else body.innerHTML = `<div class="note">Live run complete. Artifact written to the isolated live namespace.</div>`;
  p.appendChild(body);
}

// ── Replay reveals (one per stage) ──────────────────────────────────────────
const SCOPE_ORDER = ["none", "direct", "direct-plus-context", "direct-plus-temporal",
  "direct-plus-monetary", "quasi", "all", "direct-nondeterministic"];

// Cache the domain's journey so each stage reveal can show ITS concrete input→output slice.
let journeyCache = { domain: null, data: null };
async function getJourney(domain) {
  if (journeyCache.domain === domain) return journeyCache.data;
  try { journeyCache = { domain, data: await get("/journey?domain=" + encodeURIComponent(domain)) }; }
  catch (e) { journeyCache = { domain, data: null }; }
  return journeyCache.data;
}

// The concrete input→output transformation for THIS stage (the journey slice), as a clean
// input-card → arrow → output-card block. Ingest is the one that carries cleartext (highlighted
// red) turning into tokens; the rest move tokens through.
async function stageIO(stageId) {
  const j = await getJourney(currentDomain);
  const st = j && j.stages && j.stages.find(x => x.id === stageId);
  if (!st) return "";
  const isIngest = stageId === "ingest";
  const inHtml = isIngest ? markClearPii(st.in) : markTokens(esc(st.in));
  const outHtml = markTokens(esc(st.out));
  return `<div class="xform">
    <div class="xcard ${isIngest ? "in-clear" : ""}"><div class="xlab">${esc(st.in_label)}</div><div class="xval">${inHtml}</div></div>
    <div class="xarrow" aria-hidden="true"><span></span></div>
    <div class="xcard ${isIngest ? "out-prot" : ""}"><div class="xlab">${esc(st.out_label)}</div><div class="xval">${outHtml}</div></div>
  </div>${st.caption ? `<div class="xcaption">${esc(st.caption)}</div>` : ""}`;
}

// The measurement section's label per stage — names the specific thing measured, so the second
// section isn't a generic "measured result" but tells the reader what number they're about to see.
function measureLabel(stageId) {
  return ({
    ingest: "What It Cost: Protection Coverage",
    train: "What It Cost: Classifier Utility",
    retrieve: "What It Cost: Retrieval (Semantic Erasure)",
    infer: "What It Cost: Investigation Quality",
    egress: "What It Cost: Queue Precision & Leak-Check",
    present: "Who Sees What: Role-Gated Re-Identification",
    safe_harbor: "Coverage: HIPAA Identifiers Removed",
    expert_determination: "Residual Risk: Before vs After",
    gate1: "Coverage: PII Protected At The Door",
    gate2: "Who Sees What: Role Dual-Gate",
  })[stageId] || "Measured Result";
}

// Stages that have a genuine RECORD-LEVEL transformation to show (clear→tokens, prompt→reply,
// reply→verdict, record→de-identified, message→tokenized). The others are measurement-only: their
// "input→output" would just restate the metric they already show (train's score, retrieve's
// survives/collapses, the risk table, the role cards), so we omit the IO block to avoid duplication.
const IO_STAGES = new Set(["ingest", "infer", "egress", "safe_harbor", "gate1"]);

async function revealStage(s, p) {
  const wrap = el("div", "stage-reveal");
  const io = IO_STAGES.has(s.id) ? await stageIO(s.id) : "";
  // No in-panel progress track: the chip stepper above already shows position, so a second
  // indicator here was redundant.
  wrap.innerHTML =
    (io ? `<div class="sec-label">What Happens To The Data</div>${io}
        <div class="io-divider"></div><div class="sec-label">${esc(measureLabel(s.id))}</div>`
        : `<div class="sec-label">${esc(measureLabel(s.id))}</div>`);
  const body = el("div");
  wrap.appendChild(body);
  try {
    if (s.id === "train") await revealTrain(body);
    else if (s.id === "retrieve") await revealRetrieve(body);
    else if (s.id === "infer") await revealInfer(body);
    else if (s.id === "egress") await revealEgress(body);
    else if (s.id === "ingest") await revealIngest(body);
    else if (s.id === "safe_harbor") await revealSafeHarbor(body);
    else if (s.id === "expert_determination") await revealExpertDet(body);
    else if (s.id === "gate1") await revealGate1(body);
    else if (s.id === "gate2") await revealGate2(body);
    else if (s.id === "present") await revealPresent(body);
    else body.innerHTML = `<div class="note">${esc(s.measures)}: a local step with no separate measured result to show.</div>`;
  } catch (e) {
    body.innerHTML = `<div class="note">Could not load the results for this stage: ${esc(String(e))}</div>`;
  }
  p.appendChild(wrap);
}

async function revealTrain(body) {
  const rows = await get("/artifact/training");
  const by = Object.fromEntries(rows.map(r => [r.scope, r]));
  const base = by.none && by.none.average_precision;
  if (!base) { body.innerHTML = `<div class="note">The clear-data baseline for this comparison is not available.</div>`; return; }
  let t = `<table><thead><tr><th>Scope</th><th>AP</th><th>± SD</th><th>ROC-AUC</th><th>Retained</th></tr></thead><tbody>`;
  SCOPE_ORDER.forEach(sc => { const r = by[sc]; if (!r) return;
    t += `<tr><td class="name">${esc(sc)}</td><td>${r.average_precision.toFixed(3)}</td><td>${(r.average_precision_seed_std || 0).toFixed(3)}</td><td>${r.roc_auc.toFixed(3)}</td><td class="retained">${Math.round(r.average_precision / base * 100)}%</td></tr>`; });
  t += `</tbody></table>`;
  // Scope selector over the plots: each trained scope has its own SHAP + PR plot, so the viewer can
  // see how feature reliance shifts as more is tokenized (the point of the whole curve).
  const plotScopes = SCOPE_ORDER.filter(sc => by[sc]);
  const pills = plotScopes.map(sc =>
    `<button class="scope-pill${sc === "none" ? " on" : ""}" data-scope="${esc(sc)}">${esc(sc)}</button>`).join("");
  body.innerHTML = t
    + `<div class="plot-scope-bar"><span class="plot-scope-label">Plots for scope</span><div class="scope-pills" id="shapScopes">${pills}</div></div>`
    + `<div class="plots" id="shapPlots">${plot("none", "shap_beeswarm", "SHAP Feature Reliance")}${plot("none", "precision_recall", "Precision–Recall Curve")}</div>`
    + `<div class="note" id="shapNote">${scopeTrainNote("none", by.none, base)}</div>`;
  body.querySelectorAll("#shapScopes .scope-pill").forEach(b => b.onclick = () => {
    body.querySelectorAll("#shapScopes .scope-pill").forEach(x => x.classList.toggle("on", x === b));
    const sc = b.dataset.scope;
    $("#shapPlots").innerHTML = plot(sc, "shap_beeswarm", "SHAP Feature Reliance") + plot(sc, "precision_recall", "Precision–Recall Curve");
    $("#shapNote").innerHTML = scopeTrainNote(sc, by[sc], base);   // note follows the selected scope
  });
}

// A note that describes what THIS scope's plots show, grounded in the scope's real numbers — so it
// changes as you switch scope instead of a single generic line.
function scopeTrainNote(scope, r, base) {
  const ap = r ? r.average_precision : null;
  const retained = (r && base) ? Math.round(ap / base * 100) : null;
  const what = {
    "none": "The clear-data baseline: no tokenization, every feature in the clear. This is the 100% mark the other scopes are measured against.",
    "direct": "Names, addresses, emails and IDs are tokenized. The classifier leans on graph-walk and temporal features, not identity, so accuracy is unchanged from clear.",
    "direct-plus-context": "Direct identifiers plus contextual fields tokenized: still no effect, because identity was never the signal the model used.",
    "direct-plus-temporal": "Direct identifiers plus dates tokenized. Here the score is even slightly above baseline, within seed noise rather than a real gain.",
    "direct-plus-monetary": "Direct identifiers plus AMOUNT tokenized. This is the one scope that costs a few points, because the model does use transaction magnitude: watch the amount feature fall in the beeswarm.",
    "quasi": "Quasi-identifiers tokenized (amounts and dates). Behavioural graph features carry the signal, so accuracy holds near baseline.",
    "all": "Everything protectable is tokenized. The model still ranks the queue on graph structure and behaviour alone, so accuracy is essentially unchanged from clear.",
    "direct-nondeterministic": "Direct identifiers tokenized with rotating initialization vectors (an ablation): the same protection with per-value randomness.",
  }[scope] || "Feature reliance for this protection scope.";
  const retLine = retained != null
    ? ` Average precision here is <b>${ap.toFixed(3)}</b>, which is <b>${retained}%</b> of the clear baseline.`
    : "";
  return `${what}${retLine} Plots produced by the real training run.`;
}

async function revealRetrieve(body) {
  const e = await get("/artifact/erasure");
  const fp = e.direct && e.direct.identity_fisher_p_vs_baseline;
  const fpTxt = (typeof fp === "number") ? fp.toExponential(1).replace("e", " × 10^") : "n/a";
  body.innerHTML = metrics([
    metric(esc(e.none.behavioural_found + " → " + (e.direct.behavioural_found ?? "?")), "Behavioural Recall: Survives"),
    metric(esc(e.none.identity_found + " → " + (e.direct.identity_found ?? "?")), "Identity Recall: Collapses"),
    metric(esc(fpTxt), "Fisher Exact p (identity)"),
  ]) + `<div class="note">Retrieval survives or dies by what the query asks for. A token carries no embedding relationship to its plaintext, so identity search collapses while behavioural search stays intact. This is the sharpest, most reproducible result in the suite.</div>`;
}

async function revealInfer(body) {
  const c = await get("/eval-curve");
  let t = `<table><thead><tr><th>Scope</th><th>Mean Score</th><th>Model</th></tr></thead><tbody>`;
  SCOPE_ORDER.forEach(sc => { const r = c[sc]; if (!r) return;
    t += `<tr><td class="name">${esc(sc)}</td><td>${(r.mean || 0).toFixed(3)}</td><td class="name">${esc((r.models_used || ["?"])[0].split("/").pop())}</td></tr>`; });
  body.innerHTML = t + `</tbody></table><div class="note">Investigation quality across protection scopes, single-model verified (no fallback mixing). A bounded null: the model reasons about as well over tokens as over cleartext.</div>`;
}

async function revealEgress(body) {
  const h = await get("/artifact/hybrid_none");
  body.innerHTML = metrics([
    metric(esc(String(h.precision_at_capacity)), "Queue Precision @ Capacity"),
    metric(esc(String(h.queue_length)), "Alerts Ranked"),
    metric(esc(String(h.distinct_subjects_in_head)) + "/25", "Distinct Subjects In Head"),
    metric(esc(String(h.egress_blocked)), "Responses Blocked By Guard"),
  ]) + `<div class="note">Every model response is scanned before a human sees it. Clear and protected queues rank near-identically, so protecting identities does not degrade the triage.</div>`;
}

async function revealIngest(body) {
  const m = await get("/artifact/ingest");
  body.innerHTML = metrics([
    metric(esc(String(m.scopes_completed.length)), "Protection Scopes"),
    metric(esc(String(m.total_noops)), "Unprotected Values (must be 0)"),
    metric(esc(String(m.total_failures)), "Redacted (non-ISO dates)"),
  ]) + `<div class="note">The same corpus tokenized to ${esc(String(m.scopes_completed.length))} protection levels. Structure stays invariant; PII is protected. Where tokenization returns a value unchanged, that value is caught and redacted rather than trusted, so nothing passes through in the clear.</div>`;
}

async function revealSafeHarbor(body) {
  const d = await get("/artifact/healthcare_deid");
  const sh = d.safe_harbor || {};
  const present = sh.identifiers_present || {};
  const presentTxt = Object.entries(present).map(([k, v]) => `${titleCase(k.replace(/_/g, " "))} (${(v || []).join("/")})`).join("; ");
  const nCat = Object.keys(present).length;
  body.innerHTML = metrics([
    metric(esc(String(sh.n_records)), "Patient Records De-Identified"),
    metric(esc(String(nCat)) + "/18", "HIPAA Identifier Categories Present"),
    metric(esc(String(sh.noop_names_redacted)), "Unprotected Names Redacted"),
  ]) + `<div class="note">${esc(sh.name_handling || "names removed / tokenized")}. Safe-Harbor identifier categories present in this schema: ${esc(presentTxt || "none")}.</div>`;
}

async function revealExpertDet(body) {
  const d = await get("/artifact/healthcare_deid");
  const ed = d.expert_determination || {}, b = ed.before || {}, a = ed.after || {};
  const met = ed.expert_determination_met;
  // The three standard HIPAA Expert-Determination attacker models, each with a plain-language gloss
  // so a clinical/executive reader doesn't need to know the field jargon. k-anonymity is the
  // structural metric behind them.
  let t = `<table><thead><tr><th>Attacker Model</th><th>What They Assume</th><th>Before</th><th>After k=5</th></tr></thead><tbody>`;
  [["Prosecutor", "knows the person is in the data (worst-case)", "prosecutor_risk"],
   ["Journalist", "unsure if the person is in the data; uses outside info (targeted)", "journalist_risk"],
   ["Marketer", "re-identifies as many people as possible (average-case)", "marketer_risk"],
   ["k-Anonymity", "how many records each person is indistinguishable among", "k_anonymity"]]
    .forEach(([lbl, desc, k]) => {
      t += `<tr><td class="name">${esc(lbl)}</td><td class="name" style="color:var(--ink-soft)">${esc(desc)}</td><td>${esc(fmtNum(b[k]))}</td><td>${esc(fmtNum(a[k]))}</td></tr>`;
    });
  t += `</tbody></table>`;
  const ka = ed.k_anonymization || {};
  body.innerHTML = t
    + `<div class="verdict ${met ? "good" : "warn"}">${met ? "✓ Expert Determination Met" : "⚠ Expert Determination Not Certified"} On This Sample</div>`
    + `<div class="note">${met ? "Residual risk dropped below the bound." : "Worst-case (prosecutor) risk did not drop enough: more generalization or fewer quasi-identifiers would be needed. Reported honestly, not glossed."} Information loss ${esc(fmtNum(ka.information_loss))}, ${esc(String(ka.suppressed_count))} rows suppressed.</div>`;
}

// ── Role-differentiated views: the SAME protected record, re-identified per role ────────────────
// Render a role's `sees` text: masked spans ([TYPE: masked]) as warn chips, still-tagged spans as
// token chips, everything else (clear, when the role is fully entitled) as plain ink.
function renderRoleSees(text) {
  let s = esc(text);
  s = s.replace(/\[([A-Z_]+): masked\]/g, m => `<span class="masked-chip">${m}</span>`);
  s = s.replace(/\[([A-Z_]+)\][^\[]*?\[\/\1\]/g, m => `<span class="tag">${m}</span>`);
  return s;
}
// Human-readable reveal list: dedupe synonymous entity types (MED_REC / MEDICAL_RECORD_NUMBER) and
// tidy the labels, so a role card reads cleanly ("person, medical record") not "med rec, medical
// record number".
function prettyReveals(types) {
  if (!types || !types.length) return "nothing";
  const nice = { MED_REC: "medical record", MEDICAL_RECORD_NUMBER: "medical record",
    SOCIAL_SECURITY_ID: "ssn", ACCOUNT_NUMBER: "account", BANK_ACCOUNT: "bank account",
    CREDIT_CARD: "card", EMAIL_ADDRESS: "email", PHONE_NUMBER: "phone" };
  const seen = new Set();
  return types.map(t => nice[t] || t.toLowerCase().replace(/_/g, " "))
    .filter(x => !seen.has(x) && seen.add(x)).join(", ");
}
function roleCard(rv) {
  const cls = rv.fully ? "role-full" : (rv.revealed > 0 ? "role-part" : "role-masked");
  const reveals = rv.fully ? "everything" : prettyReveals(rv.may_unprotect);
  return `<div class="cell rolecard ${cls}">
    <div class="role-head"><span class="role-name">${esc(rv.label)}</span>
      <span class="role-meta">reveals ${esc(reveals)} · ${rv.revealed} shown / ${rv.withheld} withheld</span></div>
    <div class="role-sees">${renderRoleSees(rv.sees)}</div></div>`;
}
// Grid sized to the number of roles (2 or 3).
function roleGrid(views) {
  return `<div class="role-grid cols-${views.length}">${views.map(roleCard).join("")}</div>`;
}

async function revealPresent(body) {
  const j = await getJourney(currentDomain);
  const views = (j && j.role_views) || [];
  if (!views.length) { body.innerHTML = `<div class="note">No role views available for this domain.</div>`; return; }
  body.innerHTML = `<div class="note">The single presentation boundary: the same protected record, re-identified for each role. Plaintext appears only for the role entitled to it.</div>`
    + roleGrid(views)
    + `<div class="footnote">Which identifiers each role can reveal is enforced by policy: the same gate that governs access here can be centrally managed in Protegrity.</div>`;
}

async function revealGate1(body) {
  // The message → tokenized transformation is already shown in the "What Happens To The Data"
  // block above (stageIO), so here we show ONLY the measurement (count), not repeat the message.
  const d = await get("/artifact/support_gates");
  body.innerHTML = metrics([metric(esc(String(d.entities_protected)), "PII Entities Tokenized")])
    + `<div class="note">Every PII span is protected before anything downstream sees it. The tokenized message above is all the model, logs, and audit records ever receive.</div>`;
}

async function revealGate2(body) {
  const d = await get("/artifact/support_gates");
  const ag = (d.roles || []).find(r => r.name === "support_agent") || {}, su = (d.roles || []).find(r => r.name === "supervisor") || {};
  body.innerHTML = `<div class="note">The same protected reply, re-identified under two roles:</div>
    <div class="compare">
      <div class="cell role-masked"><div class="k">Support Agent · Revealed ${esc(String(ag.revealed))}</div><div class="txt-masked">${esc(ag.sees)}</div></div>
      <div class="cell role-full"><div class="k">Supervisor · Revealed ${esc(String(su.revealed))}</div><div class="txt-clear">${esc(su.sees)}</div></div>
    </div><div class="footnote">${esc(d.caveat)}</div>`;
}

// ── Assistant (chatbot): two panels read together — Conversation + Protection Boundary ──────────
// (No fake tabs, no per-turn role toggle: a free-text reply written over tokens barely differs by
// role, so a live toggle was a weak signal. Instead the role story is concentrated in ONE strong
// moment — a "same record, three roles" comparison shown once, below the conversation.) The turn's
// reply is always the protected view; the boundary panel shows why it is safe.
const CHAT_ROLE = "investigator";   // the transcript was generated for the entitled role
function renderChat() {
  // Live is only offered where the server supports it (AML in this edition). If the user is in Live
  // mode on a domain that doesn't, render a clear unavailable state with a path forward instead of
  // an input box that would 503 on send.
  const liveBlocked = mode === "live" && !domainSupportsLive(currentDomain);
  const isLive = mode === "live" && !liveBlocked;
  $("#chatbot").innerHTML = `
    <div class="chat">
      <div class="col">
        <div class="panel-label">Conversation <span class="hint">protected reply, written over tokens</span></div>
        <div class="msgs" id="msgs"><div class="footnote">${liveBlocked
          ? "Live chat is not enabled for this domain. Use Replay here to step through a recorded protected conversation, or switch to a domain with live chat enabled."
          : isLive
          ? "Live mode: each turn protects your PII, reasons over tokens, scans egress, and re-identifies by role (~$0.01/turn)."
          : "Replay mode: a committed multi-turn run over the tokenized corpus. Step through it below."}</div></div>
        ${liveBlocked
          ? `<div class="chatin"><button class="btn" id="cusereplay">Use Replay for this domain</button></div>`
          : isLive
          ? `<div class="chatin"><input id="cin" placeholder="Ask about a subject, e.g. 'Summarize the case notes for Sana Choudhury'"/><button class="btn" id="csend">Send</button></div>`
          : `<div class="replay-controls">
               <button class="btn ghost" id="cprev">‹ Prev</button>
               <button class="btn" id="cnext">Next Turn ›</button>
               <div class="turn-dots" id="cdots"></div>
               <label class="auto-toggle"><input type="checkbox" id="cauto"/> Auto-play</label>
               <span class="footnote" id="cstatus"></span>
             </div>`}
      </div>
      <div class="col">
        <div class="panel-label">Protection Boundary <span class="hint">the same turn, protected end to end</span></div>
        <div class="msgs steps" id="ints"><div class="footnote">${isLive ? "Send a message" : "Step through the transcript"} to see the per-turn boundary: tokenized in → model over tokens → egress scan → role-gated reveal.</div></div>
      </div>
    </div>
    <div id="roleFinale"></div>`;
  getJourney(currentDomain).then(() => {
    if (liveBlocked) {
      // One-click path to the mode that DOES work for this domain.
      $("#roleFinale").innerHTML = "";
      const b = $("#cusereplay"); if (b) b.onclick = () => setMode("replay");
    } else if (isLive) {
      // No turn yet — hint that the card appears after a turn (it fills in on send).
      $("#roleFinale").innerHTML = `<div class="finale"><div class="panel-label">Who Sees What <span class="hint">appears after a turn</span></div></div>`;
      $("#csend").onclick = sendTurn;
      $("#cin").onkeydown = e => { if (e.key === "Enter") sendTurn(); };
    } else {
      setupReplay();   // gotoTurn(0) renders the finale for the first turn
    }
  });
}

// The single, strong role moment: the SAME record re-identified for each of the domain's roles —
// the same "who sees what" comparison as the batch Present stage, shown once beneath the chat so
// the role story lands concretely (rather than a per-turn toggle that a token-light reply can't show).
// "Who Sees What" for the CURRENT turn: the turn's inbound message (which always carries the
// subject's PII) re-identified per role. Updates on every Next/Prev, so each turn shows how that
// message would be presented to each role. (We use the inbound, not the reply, because a free-text
// reply often has 0-2 token spans, giving no visible per-role difference; the inbound always does.)
async function renderRoleFinale(turn) {
  const host = $("#roleFinale"); if (!host) return;
  const j = await getJourney(currentDomain);
  const roleDefs = (j && j.role_views) || [];
  if (!roleDefs.length || !turn) { host.innerHTML = ""; return; }
  const views = roleDefs.map(rd => turnRoleView(turn, rd));
  host.innerHTML = `
    <div class="finale">
      <div class="panel-label">Who Sees What <span class="hint">this turn's message, re-identified per role at the presentation boundary</span></div>
      ${roleGrid(views)}
      <div class="footnote">Which identifiers each role can reveal is enforced by policy: the same gate that governs access here can be centrally managed in Protegrity.</div>
    </div>`;
}

// Build a role view for THIS turn's inbound message: the fully-entitled role sees the clear text
// (turn.user); a restricted role sees the tokenized inbound with types it may not unprotect masked.
// Align the tokenized input with the cleartext the user typed, so a partial-reveal role can show
// the REAL value for the identifiers it may see (not the internal token). The literal text around
// the tokens is identical in both strings; each [TYPE]token[/TYPE] span in `protectedInput` lines up
// with the cleartext slice at the same position in `clearInput`. We walk the tokenized string,
// copying literals through and, at each token, consuming the matching cleartext up to the next
// literal — revealing it (role may see this type) or masking it (role may not).
function turnRoleView(turn, roleDef) {
  const allow = new Set(roleDef.may_unprotect || []);
  const protectedInput = (turn.internals && turn.internals.protected_input) || "";
  const clearInput = turn.user || "";
  let revealed = 0, withheld = 0;
  if (roleDef.fully) {
    (protectedInput.match(/\[([A-Z_]+)\][^\[]*?\[\/\1\]/g) || []).forEach(() => revealed++);
    return { label: roleDef.label, may_unprotect: roleDef.may_unprotect, fully: true,
      revealed, withheld: 0, sees: clearInput };
  }
  const tokenRe = /\[([A-Z_]+)\][^\[]*?\[\/\1\]/g;
  let out = "", lastEnd = 0, clearPos = 0, m;
  while ((m = tokenRe.exec(protectedInput)) !== null) {
    const literal = protectedInput.slice(lastEnd, m.index);  // shared text before this token
    out += literal;
    clearPos += literal.length;                              // same literal advances the clear cursor
    // The cleartext value spans from here to where the NEXT shared literal begins.
    const afterToken = protectedInput.slice(tokenRe.lastIndex);
    const nextLiteral = afterToken.match(/^[^[]*/)[0];        // literal that follows this token
    const nextLitStart = nextLiteral
      ? clearInput.indexOf(nextLiteral, clearPos)
      : clearInput.length;
    const clearEnd = nextLitStart >= 0 ? nextLitStart : clearInput.length;
    const clearVal = clearInput.slice(clearPos, clearEnd);
    if (allow.has(m[1])) { revealed++; out += clearVal; }     // reveal the real value
    else { withheld++; out += `[${m[1]}: masked]`; }          // mask it
    clearPos = clearEnd;
    lastEnd = tokenRe.lastIndex;
  }
  out += protectedInput.slice(lastEnd);                       // trailing shared literal
  return { label: roleDef.label, may_unprotect: roleDef.may_unprotect, fully: false,
    revealed, withheld, sees: out };
}

// The entitled role's committed view (the transcript was generated for it), for the boundary's
// role-gated step-4 line.
function entitledRoleView() {
  const j = journeyCache.data;
  const views = (j && j.role_views) || [];
  return views.find(v => v.fully) || views[views.length - 1] || null;
}

// ---- Replay mode: MANUAL step-through by default (user controls the pace), optional auto-play ----
let replayTranscript = null, replayIdx = -1, replayTimer = null;
async function setupReplay() {
  replayIdx = -1; replayTranscript = null;
  if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
  try {
    replayTranscript = await get("/chat/replay?domain=" + encodeURIComponent(currentDomain));
  } catch (e) { $("#cstatus").textContent = "No transcript for this domain."; return; }
  if (!replayTranscript || replayTranscript.detail) {
    $("#msgs").innerHTML = `<div class="footnote">No recorded conversation is available for ${esc(titleCase(currentDomain))} yet. Switch to Live to run one, or pick another domain.</div>`;
    ["cprev", "cnext", "cauto"].forEach(id => { const e = $("#" + id); if (e) e.disabled = true; });
    return;
  }
  $("#cstatus").textContent = `${replayTranscript.turns.length} turns · ${esc(replayTranscript.model)}`;
  $("#cnext").onclick = () => gotoTurn(replayIdx + 1);
  $("#cprev").onclick = () => gotoTurn(replayIdx - 1);
  $("#cauto").onchange = e => toggleAuto(e.target.checked);
  renderDots();
  gotoTurn(0);   // show the first turn immediately, then wait for the user
}

function renderDots() {
  const d = $("#cdots"); if (!d || !replayTranscript) return;
  d.innerHTML = replayTranscript.turns.map((t, i) =>
    `<button class="dot${i === replayIdx ? " on" : ""}${t.internals.egress_blocked ? " held" : ""}" data-i="${i}" title="Turn ${i + 1}"></button>`).join("");
  d.querySelectorAll(".dot").forEach(b => b.onclick = () => gotoTurn(+b.dataset.i));
}

// Render EXACTLY the turn at index i (idempotent — rebuilds the conversation up to i, so Prev works
// and jumping via a dot works). The user drives this; nothing advances on its own unless auto-play
// is on (and auto-play is a slow, clearly-optional convenience).
function gotoTurn(i) {
  if (!replayTranscript) return;
  const n = replayTranscript.turns.length;
  if (i < 0 || i >= n) { if (i >= n) stopAuto(); return; }
  replayIdx = i;
  const m = $("#msgs"); m.innerHTML = "";
  const turns = replayTranscript.turns;
  // Render turns 0..i, but COLLAPSE a run of consecutive held turns into ONE summary line instead of
  // repeating the identical "reply held" bubble. A passed turn (or the current turn) always renders
  // in full so the conversation stays readable.
  for (let k = 0; k <= i;) {
    const t = turns[k];
    if (t.internals.egress_blocked && !(k === i)) {
      // start of a held run — extend it as far as it stays held AND doesn't include the current turn
      let j = k;
      while (j + 1 <= i && turns[j + 1].internals.egress_blocked && (j + 1) !== i) j++;
      if (j > k) {
        m.appendChild(el("div", "msg held-run",
          `<div class="who">Egress boundary</div><div class="b">${j - k + 1} turns held before release: the guard withheld each reply. See the Protection Boundary panel for the scan on the current turn.</div>`));
        k = j + 1;
        continue;
      }
    }
    const cur = k === i;
    m.appendChild(el("div", "msg user" + (cur ? " cur" : ""), `<div class="who">You</div><div class="b">${esc(t.user)}</div>`));
    const held = t.internals.egress_blocked;
    const bodyHtml = held
      ? `<em>${esc("Reply held at the egress boundary. See the Protection Boundary panel.")}</em>`
      : mdLite(markTokens(deEmDash(esc(t.reply_over_tokens || "(no reply)"))));
    m.appendChild(el("div", "msg bot" + (cur ? " cur" : ""), `<div class="who">Aegis · over tokens ${egressPill(t.internals)}</div><div class="b">${bodyHtml}</div>`));
    k++;
  }
  // scroll the current turn into view (not the very bottom), so the user reads from its start
  const curEls = m.querySelectorAll(".cur");
  if (curEls.length) curEls[0].scrollIntoView({ block: "start", behavior: "smooth" });
  renderInternals(replayTranscript.turns[i].internals, { reply: "" }, replayTranscript.turns[i]);
  renderRoleFinale(replayTranscript.turns[i]);   // "Who Sees What" updates to this turn's message
  $("#cstatus").textContent = `Turn ${i + 1} / ${n}`;
  $("#cprev").disabled = i === 0;
  $("#cnext").disabled = i === n - 1;
  renderDots();
}

function toggleAuto(on) {
  if (on) {
    // slow, readable cadence; stops at the end. The user can uncheck any time.
    replayTimer = setInterval(() => {
      if (replayIdx >= replayTranscript.turns.length - 1) { stopAuto(); return; }
      gotoTurn(replayIdx + 1);
    }, 4200);
  } else { stopAuto(); }
}
function stopAuto() {
  if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
  const cb = $("#cauto"); if (cb) cb.checked = false;
}

// ---- Live mode ----
async function sendTurn() {
  const inp = $("#cin"), msg = inp.value.trim(); if (!msg) return;
  inp.value = "";
  const m = $("#msgs");
  m.appendChild(el("div", "msg user", `<div class="who">You</div><div class="b">${esc(msg)}</div>`));
  const wait = el("div", "msg bot", `<div class="who">Aegis</div><div class="b"><span class="running">Protecting → Reasoning → Scanning → Re-Identifying</span></div>`);
  m.appendChild(wait); m.scrollTop = m.scrollHeight;
  try {
    const headers = { "content-type": "application/json" };
    if (window.AMLGUARD_UI_TOKEN) headers["X-AMLGuard-Token"] = window.AMLGUARD_UI_TOKEN;
    const resp = await fetch(API + "/chat/turn", {
      method: "POST", headers,
      body: JSON.stringify({ message: msg, domain: currentDomain, role: CHAT_ROLE, conversation_id: "ui-demo" })
    });
    const r = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      // The API sends a clean, human-readable reason in `detail` (e.g. a 503 "live chat is
      // corpus-not-loaded message, or a 429 turn-cap). Show THAT, not a broken empty reply.
      wait.className = "msg bot err";
      wait.querySelector(".b").textContent = r.detail || `Request failed (${resp.status}).`;
      m.scrollTop = m.scrollHeight;
      return;
    }
    wait.querySelector(".b").innerHTML = mdLite(markTokens(deEmDash(esc(r.reply || "(no reply)"))));
    renderInternals(r.internals, r);
    // "Who Sees What" for the live turn: build a pseudo-turn (clear user msg + tokenized inbound).
    renderRoleFinale({ user: msg, internals: r.internals });
  } catch (e) {
    wait.className = "msg bot err";
    wait.querySelector(".b").textContent = "Could not reach the assistant: " + e;
  }
  m.scrollTop = m.scrollHeight;
}

function egressPill(i) {
  if (i.egress_blocked) return `<span class="pill hold">Held</span>`;
  const d = i.egress_detail || {};
  if (d.outcome === "rejected") return `<span class="pill disc">Discounted</span>`;
  return `<span class="pill pass">Passed</span>`;
}

// A guardrail score is always shown to 2 decimals, so a genuine 0.00 reads as a real score (a clean
// reply) rather than a bare "0" that looks broken or missing.
const fmtScore = v => {
  const n = Number(v);
  return (v === null || v === undefined || v === "" || Number.isNaN(n)) ? String(v ?? "") : n.toFixed(2);
};
function egressDetail(i) {
  const d = i.egress_detail; if (!d || !d.processors) return "";
  let s = "";
  d.processors.forEach(p => { s += ` ${esc(p.name)} <b>${esc(fmtScore(p.score))}</b>`; });
  // Always show the conversation-risk score when the field is present. A genuine 0.00 is a real
  // value (a clean reply), not a missing one, so it must not be hidden by a falsy check.
  if (d.batch_score !== undefined && d.batch_score !== null) s += ` · conversation risk <b>${esc(fmtScore(d.batch_score))}</b>`;
  return s;
}

function renderInternals(i, r, turn) {
  const box = $("#ints"); if (!i) { box.innerHTML = ""; return; }
  const tokd = esc(i.protected_input).replace(/\[([A-Z_]+)\][^\[]+\[\/\1\]/g, m => `<span class="tag">${m}</span>`);
  // Step 4: the transcript was generated for the entitled role; the full "who sees what" per-role
  // comparison lives in the Who-Sees-What card below the conversation.
  const rv = entitledRoleView();
  const reidTxt = rv
    ? `${rv.label}: fully re-identified (${rv.revealed} identifiers). See “Who Sees What” below for the other roles`
    : `${turn ? turn.would_reidentify : i.revealed} re-identified for the entitled role`;
  // Describe what the egress scan DOES, not the raw processor name. (The injection processor is
  // named "customer-support" across domains — a Protegrity config detail — so showing it verbatim
  // read as a wrong-domain "support model" on the AML chatbot.)
  const guard = "prompt-injection + PII scan";
  box.innerHTML = `
    <div class="row"><div class="k"><span class="n">1</span>Inbound Tokenized · ${esc(String(i.entities_protected))} PII</div><div class="body">${tokd}</div></div>
    <div class="row"><div class="k"><span class="n">2</span>What The Model Saw · Tokens Only</div><div class="body">${tokd}</div></div>
    <div class="row"><div class="k"><span class="n">3</span>Egress Scan · ${esc(guard)} ${egressPill(i)}</div><div class="body">${egressDetail(i)}</div></div>
    <div class="row"><div class="k"><span class="n">4</span>Presentation · Role-Gated</div><div class="body">${esc(reidTxt)}</div></div>`;
}

boot();
