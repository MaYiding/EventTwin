/* 企业外部情报系统 —— 前端主逻辑（无依赖，原生 JS） */
"use strict";

/* ---------------- 基础工具 ---------------- */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtTime = ts => (ts || "").replace("T", " ").slice(5, 19);
const short = (s, n = 40) => (s || "").length > n ? s.slice(0, n) + "…" : (s || "");

function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = "toast " + kind;
  t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), 3200);
}

async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(data.error || r.statusText), { data, status: r.status });
  return data;
}

/* ---------------- 类型字典 ---------------- */
const TYPE_LABEL = {
  product_launch: "产品发布", price_change: "价格调整", patent_publication: "专利公开",
  merger_deal: "并购交易", partnership: "合作签约", executive_change: "人事变动",
  financial_report: "财报发布", sales_report: "销量快报", investment: "投融资",
  market_entry: "市场进入", legal_action: "诉讼监管", incident: "事故负面",
  award: "获奖荣誉", policy_change: "战略调整", expansion: "产能扩张", other: "其他",
};
const ACTION_LABEL = {
  attach: "规则归并", create_provisional: "新建事件", judge_attach: "判别归并",
  judge_create: "判别新建", pending: "待判", manual: "人工", reject: "撤回",
};
const ACTION_CLASS = {
  attach: "green", create_provisional: "blue", judge_attach: "green",
  judge_create: "blue", pending: "orange", manual: "violet", reject: "red",
};
const CHANGE_LABEL = {
  state_transition: "状态变化", correction: "官方更正", conflict: "事实争议",
  retraction: "撤回", first_seen: "首次发现", expire: "过期",
};
const CHANGE_CLASS = {
  state_transition: "blue", correction: "violet", conflict: "red",
  retraction: "red", first_seen: "green", expire: "gray",
};
const STAGE_COLOR = {
  ingest: "#2563eb", parse: "#0e9bb5", extract: "#d98207", entity: "#0f9d6e",
  vector: "#7c5cf0", recall: "#0e9bb5", resolve: "#dc3d43", assertion: "#0f9d6e",
  relation: "#7c5cf0", change: "#d98207", project: "#4b5567", pipeline: "#101529",
  crud: "#7c5cf0", qa: "#2563eb", server: "#4b5567", replay: "#2563eb",
};

/* ---------------- 视图切换 ---------------- */
const VIEW_TITLE = { overview: "概览", feed: "实时流水", decisions: "归类决策",
  graph: "知识图谱", facts: "事实与断言", changes: "变化卡", qa: "带证据问答" };
$$(".nav-item").forEach(btn => btn.addEventListener("click", () => {
  $$(".nav-item").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  const v = btn.dataset.view;
  $$(".view").forEach(x => x.classList.remove("active"));
  $("#view-" + v).classList.add("active");
  $("#crumbCur").textContent = VIEW_TITLE[v] || v;
  if (v === "graph") { graph.ensure(); const gm = $("#graphMode"); if (gm && !gm.value) gm.value = "aggregate"; $("#graphFocus").disabled = true; }
  if (v === "facts") facts.load();
  if (v === "changes") changes.load();
  if (v === "decisions") decisions.load();
}));

/* ---------------- SSE 实时流 ---------------- */
const feed = {
  paused: false, buffer: [], max: 600, es: null, connected: false,
  stageFilter: "", levelFilter: "",
  connect() {
    if (this.es) this.es.close();
    const lastId = +(localStorage.getItem("feedCursor") || 0);
    this.es = new EventSource(`/api/stream?after_id=${Math.max(0, lastId - 50)}`);
    this.es.onopen = () => {
      this.connected = true;
      const d = $("#liveDot");
      d.classList.add("on");
      d.lastElementChild.textContent = "实时流已连接";
    };
    this.es.onerror = () => {
      this.connected = false;
      const d = $("#liveDot");
      d.classList.remove("on");
      d.lastElementChild.textContent = "实时流连接中…";
    };
    this.es.onmessage = ev => {
      if (ev.data === "[DONE]") return;
      try { this.push(JSON.parse(ev.data)); } catch (_) { /* 忽略坏帧 */ }
      localStorage.setItem("feedCursor", this.lastId || 0);
    };
  },
  get lastId() { return this.buffer.length ? this.buffer[this.buffer.length - 1].event_id : 0; },
  push(item) {
    this.buffer.push(item);
    if (this.buffer.length > this.max) this.buffer.splice(0, this.buffer.length - this.max);
    if (!this.paused) this.render(item);
    // 阶段筛选项按需补充
    const sel = $("#feedStage");
    if (sel && ![...sel.options].some(o => o.value === item.stage)) {
      const opt = document.createElement("option");
      opt.value = item.stage;
      opt.textContent = item.stage;
      sel.appendChild(opt);
    }
    overview.refreshSoon();
  },
  visible(item) {
    if (this.stageFilter && item.stage !== this.stageFilter) return false;
    if (this.levelFilter && item.level !== this.levelFilter) return false;
    return true;
  },
  render(item) {
    if (!this.visible(item)) return;
    const el = document.createElement("div");
    el.className = "feed-item " + item.level;
    const color = STAGE_COLOR[item.stage] || "#4b5567";
    let meta = "";
    try {
      const d = JSON.parse(item.data_json || "{}");
      const bits = [];
      if (d.cluster_id) bits.push("cluster " + String(d.cluster_id).slice(0, 8));
      if (d.action) bits.push(ACTION_LABEL[d.action] || d.action);
      if (d.reason) bits.push("理由: " + d.reason);
      if (d.best_score != null) bits.push("最高分 " + d.best_score);
      if (d.decision) bits.push("决定: " + d.decision);
      if (d.duplicate) bits.push("重复内容");
      meta = bits.join(" · ");
    } catch (_) { /* 空数据 */ }
    el.innerHTML = `<span class="t">${fmtTime(item.ts)}</span>
      <span class="stage-chip" style="background:${color}1a;color:${color}">${esc(item.stage)}</span>
      <div class="m"><div class="msg">${esc(item.message)}</div>
      ${meta ? `<div class="meta">${esc(meta)}</div>` : ""}</div>`;
    const list = $("#feedList");
    const atBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 80;
    list.appendChild(el);
    while (list.children.length > this.max) list.firstChild.remove();
    if (atBottom) list.scrollTop = list.scrollHeight;
    $("#feedCount").textContent = list.children.length + " 条";
  },
  rerenderAll() {
    $("#feedList").innerHTML = "";
    $("#feedCount").textContent = "0 条";
    this.buffer.forEach(i => this.render(i));
  },
};
$("#feedPause").addEventListener("click", e => {
  feed.paused = !feed.paused;
  e.target.textContent = feed.paused ? "继续" : "暂停";
});
$("#feedClear").addEventListener("click", () => { $("#feedList").innerHTML = ""; });
$("#feedStage").addEventListener("change", e => { feed.stageFilter = e.target.value; feed.rerenderAll(); });
$("#feedLevel").addEventListener("change", e => { feed.levelFilter = e.target.value; feed.rerenderAll(); });

/* ---------------- 概览 ---------------- */
const overview = {
  timer: null, stats: null,
  async refresh() {
    try {
      const { stats, queues } = await api("/api/stats");
      this.stats = stats;
      this.renderStats(stats, queues);
      this.renderStages();
    } catch (e) { /* 服务未就绪 */ }
    try {
      const h = await api("/api/health");
      this.renderHealth(h);
    } catch (_) { /* 忽略 */ }
    try {
      const { manifest } = await api("/api/manifest");
      if (manifest) this.renderManifest(manifest);
    } catch (_) { /* 忽略 */ }
  },
  refreshSoon() {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.refresh(), 800);
  },
  renderStats(s, q) {
    const qPending = Object.values(q).reduce((a, x) => a + (x.pending || 0) + (x.running || 0), 0);
    const qDead = Object.values(q).reduce((a, x) => a + (x.dead || 0), 0);
    const cards = [
      { k: "资料文档", v: s.documents, sub: `${s.document_versions} 个版本 · 重复 ${s.duplicates}`, c: "" },
      { k: "事件提及", v: s.mentions, sub: `无效 ${s.mentions_invalid} · 待归属 ${s.pending_mentions}`, c: "blue" },
      { k: "事件簇", v: s.clusters, sub: `过程 ${s.processes} · 实体 ${s.entities}`, c: "orange" },
      { k: "断言/选择", v: s.assertions, sub: `选择历史 ${s.selections} 行`, c: "green" },
      { k: "图谱关系", v: s.relations, sub: `变化卡 ${s.changes} · 向量 ${s.vectors ?? 0}`, c: "violet" },
      { k: "队列积压", v: qPending, sub: qDead ? `死信 ${qDead} ⚠` : "死信 0", c: qDead ? "orange" : "" },
      { k: "自动归并", v: (s.decisions.attach || 0) + (s.decisions.judge_attach || 0),
        sub: `判别新建 ${(s.decisions.judge_create || 0) + (s.decisions.create_provisional || 0)}`, c: "green" },
      { k: "待判", v: s.decisions.pending || 0, sub: "灰区/证据不足", c: (s.decisions.pending || 0) > 0 ? "orange" : "" },
    ];
    $("#statCards").innerHTML = cards.map(c => `
      <div class="card stat"><div class="k">${c.k}</div>
      <div class="v ${c.c}">${c.v ?? 0}</div><div class="sub">${c.sub}</div></div>`).join("");
  },
  renderStages() {
    const stages = ["ingest", "extract", "entity", "vector", "resolve", "assertion", "project"];
    const labels = { ingest: "采集入库", extract: "提及抽取", entity: "实体规范化",
      vector: "向量化", resolve: "同事件判别", assertion: "事实更新", project: "索引投影" };
    $("#stageList").innerHTML = stages.map(st => {
      const run = this._stageRuns && this._stageRuns[st];
      const color = run === "completed" ? "var(--ok)" : run === "running" ? "var(--warn)"
        : run === "failed" ? "var(--err)" : "var(--ink-3)";
      return `<div class="row" style="justify-content:space-between;padding:7px 0;
        border-bottom:1px solid #f0f2f8">
        <span><b style="color:${color}">●</b>&nbsp; ${labels[st]} <span class="tag mono"
        style="color:var(--ink-3);font-size:11px">${st}</span></span>
        <span class="badge ${run === "completed" ? "green" : run === "running" ? "orange" : "gray"}">
        ${run === "completed" ? "已完成" : run === "running" ? "进行中" : run === "failed" ? "失败" : "未开始"}</span></div>`;
    }).join("");
  },
  async renderStagesFromDB() {
    try {
      const { stages } = await api("/api/pipeline/status");
      const runs = {};
      for (const [stage, info] of Object.entries(stages)) {
        runs[stage] = info.status === "completed" ? "completed"
          : info.status === "running" ? "running"
          : info.status === "failed" ? "failed" : "idle";
      }
      this._stageRuns = runs;
      this.renderStages();
    } catch (_) { /* 忽略 */ }
  },
  renderHealth(h) {
    $("#modelHealth").innerHTML = ["chat", "embed", "jev"].map(k => {
      const it = (h.models || {})[k] || { ok: false };
      const name = { chat: "Qwen3.8-27B · 生成", embed: "Embedding-8B · 召回", jev: "Jev · 判定" }[k];
      return `<span class="badge ${it.ok ? "green" : "red"}">${it.ok ? "●" : "○"} ${name}</span>`;
    }).join("") + `<span class="badge gray">${esc(h.version || "")}</span>`;
  },
  renderManifest(m) {
    $("#manifestBox").innerHTML = [
      ["语料哈希", m.corpus_hash], ["配置哈希", m.config_hash], ["代码版本", m.code_version],
      ["LLM 缓存", m.llm_cache ? "启用（重放可复现）" : "关闭"], ["启动时间", fmtTime(m.started_at)],
    ].map(([k, v]) => `<div class="k">${k}</div><div class="mono" style="font-size:12px">${esc(v ?? "-")}</div>`).join("");
  },
};
$("#runPipeline").addEventListener("click", async e => {
  const btn = e.target;
  btn.disabled = true;
  btn.textContent = "流水线运行中…";
  try {
    await api("/api/pipeline/run", { method: "POST", body: {} });
    toast("流水线已启动，实时流水页可观测全过程", "ok");
    overview.renderStagesFromDB();
  } catch (err) {
    toast("启动失败: " + err.message, "err");
  } finally {
    setTimeout(() => { btn.disabled = false; btn.textContent = "▶ 运行流水线"; }, 4000);
  }
});

async function loadMiniFeed() {
  try {
    const { events } = await api("/api/events?limit=12");
    $("#miniFeed").innerHTML = events.map(item => {
      const color = STAGE_COLOR[item.stage] || "#4b5567";
      return `<div class="feed-item"><span class="t">${fmtTime(item.ts)}</span>
        <span class="stage-chip" style="background:${color}1a;color:${color}">${esc(item.stage)}</span>
        <div class="m"><div class="msg">${esc(item.message)}</div></div></div>`;
    }).join("") || `<div class="empty">暂无事件</div>`;
  } catch (_) { /* 忽略 */ }
}

/* ---------------- 归类决策 ---------------- */
const decisions = {
  cache: [],
  async load() {
    try {
      const { decisions: rows } = await api("/api/decisions?limit=200");
      this.cache = rows;
      this.render();
    } catch (e) { toast("决策加载失败: " + e.message, "err"); }
  },
  render() {
    const f = $("#decFilter").value;
    const rows = this.cache.filter(r => !f || r.action === f);
    const tb = $("#decTable tbody");
    tb.innerHTML = rows.map((r, i) => {
      const score = (() => {
        try { const sc = JSON.parse(r.scores_json || "{}");
          const v = Object.values(sc); return v.length ? Math.max(...v).toFixed(3) : null; }
        catch (_) { return null; }
      })();
      return `<tr>
        <td class="mono" style="font-size:11.5px;color:var(--ink-3)">${fmtTime(r.created_at)}</td>
        <td><span class="badge ${ACTION_CLASS[r.action] || "gray"}">${ACTION_LABEL[r.action] || r.action}</span></td>
        <td>${esc(short(r.frame_text, 64))}</td>
        <td>${r.cluster_summary ? esc(short(r.cluster_summary, 26)) +
          ` <span class="mono" style="color:var(--ink-3);font-size:11px">v${""}</span>` : "—"}</td>
        <td class="mono">${score ?? "—"}</td>
        <td style="color:var(--ink-2);font-size:12px">${esc(short(r.reason_code, 30))}</td>
        <td><button class="btn ghost sm" data-dec="${i}">详情</button></td></tr>`;
    }).join("") || `<tr><td colspan="7"><div class="empty">暂无决策（先运行流水线）</div></td></tr>`;
    $$("[data-dec]", tb).forEach(b => b.addEventListener("click", () => {
      const r = rows[+b.dataset.dec];
      let judge = "", cand = "";
      try { judge = JSON.stringify(JSON.parse(r.judge_json || "null"), null, 1); } catch (_) { /* 空 */ }
      try { cand = JSON.stringify(JSON.parse(r.candidates_json || "[]"), null, 1); } catch (_) { /* 空 */ }
      openModal(`归类决策 · ${ACTION_LABEL[r.action] || r.action}`, `
        <div class="kv">
          <span class="k">新提及</span><span>${esc(r.frame_text || "")}</span>
          <span class="k">目标事件</span><span>${esc(r.cluster_summary || "—")}
            <span class="mono" style="font-size:11px">${esc(r.target_cluster_id || "")}</span></span>
          <span class="k">理由码</span><span>${esc(r.reason_code || "")}</span>
          <span class="k">模型</span><span class="mono">${esc(r.model_id || "")} / ${esc(r.policy_version || "")}</span>
        </div>
        <div class="dec-detail mt"><b>候选与特征分</b><pre>${esc(cand)}</pre></div>
        ${judge && judge !== "null" ? `<div class="dec-detail mt"><b>判别器输出</b><pre>${esc(judge)}</pre></div>` : ""}`);
    }));
  },
};
$("#decFilter").addEventListener("change", () => decisions.render());
$("#decRefresh").addEventListener("click", () => decisions.load());

/* ---------------- 知识图谱（canvas 力导向） ---------------- */
const graph = {
  nodes: [], edges: [], byId: {}, sim: null, ctx: null,
  canvas: null, tf: { x: 0, y: 0, k: 1 }, dragging: null, panning: false,
  lastPos: null, selected: null, raf: null,
  COLORS: { entity: "#2563eb", event: "#f59e0b", process: "#7c5cf0", assertion: "#10b981" },
  ensure() {
    if (!this.canvas) {
      this.canvas = $("#graphCanvas");
      this.ctx = this.canvas.getContext("2d");
      this.bindEvents();
      this.bindToolbar();
      window.addEventListener("resize", () => this.resize());
    }
    this.resize();
    this.load();
  },
  resize() {
    if (!this.canvas) return;
    const r = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = r.width * dpr;
    this.canvas.height = r.height * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = r.width; this.h = r.height;
    if (this.nodes.length) { this.relax(60); this.fitView(); this.draw(); }
  },
  async load() {
    try {
      const mode = $("#graphMode") ? $("#graphMode").value : "aggregate";
      const focus = $("#graphFocus").value || null;
      let url = "/api/graph?mode=" + mode;
      if (focus && mode !== "aggregate") url += `&focus=${encodeURIComponent(focus)}&hops=2`;
      const data = await api(url);
      this.build(data, mode);
      this.loadFocusOptions();
      this.refreshSearchList();
      this.typeOff = new Set();
      document.querySelectorAll("#graphLegend .legend-chip").forEach(c => c.classList.remove("off"));
    } catch (e) { toast("图谱加载失败: " + e.message, "err"); }
  },
  async loadFocusOptions() {
    if (this._focusLoaded) return;
    try {
      const { nodes } = await api("/api/nodes/entity?limit=400");
      $("#graphFocus").innerHTML = `<option value="">全图（按重要度采样）</option>` +
        nodes.map(n => `<option value="${esc(n.id)}">${esc(n.title)}</option>`).join("");
      this._focusLoaded = true;
    } catch (_) { /* 忽略 */ }
  },
  build(data, mode) {
    const W = this.w || 900, H = this.h || 600;
    this.mode = mode || data.mode || "full";
    this.nodes = data.nodes.map((n, i) => {
      const a = (i / Math.max(data.nodes.length, 1)) * Math.PI * 2;
      const rad = Math.min(W, H) * (this.mode === "aggregate" ? 0.3 : 0.32);
      // 节点半径一次性算好，draw/tick/hit 共用（力导向按半径感知）
      const r = this.mode === "aggregate"
        ? 8 + Math.sqrt(n.n_events || 1) * 2.4
        : (n.type === "process" ? 13 : n.type === "event" ? 10 : 8);
      return { ...n, _r: r, x: W / 2 + rad * Math.cos(a) + (Math.random() - .5) * 40,
        y: H / 2 + rad * Math.sin(a) + (Math.random() - .5) * 40, vx: 0, vy: 0 };
    });
    this.byId = Object.fromEntries(this.nodes.map(n => [n.id, n]));
    this.edges = data.edges.filter(e => this.byId[e.source] && this.byId[e.target]);
    this.selected = null;
    this.settled = false;
    this.relax(250);      // 同步松弛代替 RAF 动画：后台页面/RAF 链中断不再导致画布空白
    this.settled = true;
    this.fitView();
    this.draw();
  },
  // 同步松弛 N 帧（不逐帧绘制），供 build 一次性收敛与拖拽后局部重排
  relax(iters) {
    for (let i = 0; i < iters; i++) this.tick(false);
  },
  // 自适应视野：把节点包围盒缩放平移到画布内（留边），保证"整个图"一屏可见
  fitView() {
    if (!this.nodes.length) { this.tf = { x: 0, y: 0, k: 1 }; return; }
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (const n of this.nodes) {
      x0 = Math.min(x0, n.x - n._r); x1 = Math.max(x1, n.x + n._r);
      y0 = Math.min(y0, n.y - n._r - 26); y1 = Math.max(y1, n.y + n._r + 34);
    }
    const w = this.w || 900, h = this.h || 600;
    const k = Math.max(0.3, Math.min(1.6, 0.9 * Math.min(w / (x1 - x0), h / (y1 - y0))));
    this.tf = {
      k,
      x: (w - (x0 + x1) * k) / 2,
      y: (h - (y0 + y1) * k) / 2,
    };
  },
  start() {
    // 兼容保留：外部若调用 start() → 同步收敛 + 重绘
    this.relax(120);
    this.fitView();
    this.draw();
  },
  tick(drawEach = true) {
    // 力：斥力（半径感知） + 弹簧（按两端半径定长度） + 向心力
    const N = this.nodes;
    for (let i = 0; i < N.length; i++) {
      const a = N[i];
      for (let j = i + 1; j < N.length; j++) {
        const b = N[j];
        let dx = b.x - a.x, dy = b.y - a.y;
        let d2 = dx * dx + dy * dy || 1;
        const reach = (a._r + b._r) * 2.2 + 120;
        if (d2 > reach * reach) continue;
        // 斥力强度 ∝ 两节点半径乘积（大节点推开更远）
        const f = Math.max(2600, (a._r + 8) * (b._r + 8) * 2.6) / d2;
        const d = Math.sqrt(d2);
        dx /= d; dy /= d;
        a.vx -= dx * f; a.vy -= dy * f; b.vx += dx * f; b.vy += dy * f;
      }
    }
    for (const e of this.edges) {
      const a = this.byId[e.source], b = this.byId[e.target];
      let dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const rest = a._r + b._r + 36;
      const f = (d - rest) * 0.012;
      dx /= d; dy /= d;
      a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
    }
    const cx = (this.w || 900) / 2, cy = (this.h || 600) / 2;
    for (const n of N) {
      n.vx += (cx - n.x) * 0.0025; n.vy += (cy - n.y) * 0.0025;
      if (n === this.dragging) { n.vx = n.vy = 0; continue; }
      n.vx *= 0.82; n.vy *= 0.82;
      n.x += Math.max(-8, Math.min(8, n.vx));
      n.y += Math.max(-8, Math.min(8, n.vy));
    }
    if (drawEach) this.draw();
  },
  // 关系边中文短名（层级边重点标注）
  REL_LABEL: { subsidiary_of: "子公司", brand_of: "品牌", participates_in: "参与",
               part_of_process: "属于过程", precedes: "先后", related_to: "相关",
               supports: "支持", contradicts: "矛盾", corrects: "更正" },
  HIER_REL: new Set(["subsidiary_of", "brand_of"]),
  // 焦点节点（hover/选中）的邻接集合——用于高亮淡化
  neighborSet(n) {
    if (!n) return null;
    const s = new Set([n.id]);
    for (const e of this.edges) {
      if (e.source === n.id) s.add(e.target);
      else if (e.target === n.id) s.add(e.source);
    }
    return s;
  },
  // 标签胶囊：白底圆角 + 描边 + 深色文字（比裸字清晰得多）
  labelPill(ctx, text, x, y, font, color) {
    ctx.font = font;
    const w = ctx.measureText(text).width;
    ctx.beginPath();
    ctx.roundRect(x - w / 2 - 6, y - 9, w + 12, 18, 9);
    ctx.fillStyle = "rgba(255,255,255,.94)";
    ctx.fill();
    ctx.strokeStyle = "rgba(198,206,224,.55)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(text, x, y);
    ctx.textBaseline = "alphabetic";
  },
  draw() {
    const ctx = this.ctx; if (!ctx) return;
    const W = this.w, H = this.h;
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.translate(this.tf.x, this.tf.y);
    ctx.scale(this.tf.k, this.tf.k);
    const k = this.tf.k;
    const focusNode = this.hover || this.selected;
    const nbr = this.neighborSet(focusNode);
    const dim = n => (nbr && !nbr.has(n.id)) ? 0.16 : (n.state === "provisional" ? 0.66 : 1);
    const typeOff = this.typeOff || new Set();
    // ---- 边 ----
    for (const e of this.edges) {
      const a = this.byId[e.source], b = this.byId[e.target];
      if (typeOff.has(a.type) || typeOff.has(b.type)) continue;
      const hier = this.HIER_REL.has(e.relation);
      const hi = focusNode && (e.source === focusNode.id || e.target === focusNode.id);
      const faded = nbr && !hi;
      ctx.globalAlpha = faded ? 0.12 : 1;
      ctx.strokeStyle = hi ? "#3556f5" : hier ? "#8fa7ff" : "#c4cbdb";
      ctx.lineWidth = (hi ? 2.4 : hier ? 1.8 : 1) / Math.max(k, 0.4);
      if (hier) ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      // 层级边箭头 + 标签
      if (hier && k > 0.55) {
        const ang = Math.atan2(b.y - a.y, b.x - a.x);
        const ax = b.x - Math.cos(ang) * (b._r + 4), ay = b.y - Math.sin(ang) * (b._r + 4);
        const s = 5 / Math.max(k, 0.55);
        ctx.fillStyle = hi ? "#3556f5" : "#8fa7ff";
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(ax - Math.cos(ang - 0.44) * s * 1.7, ay - Math.sin(ang - 0.44) * s * 1.7);
        ctx.lineTo(ax - Math.cos(ang + 0.44) * s * 1.7, ay - Math.sin(ang + 0.44) * s * 1.7);
        ctx.closePath(); ctx.fill();
        if (k > 0.8) {
          this.labelPill(ctx, this.REL_LABEL[e.relation] || e.relation,
            (a.x + b.x) / 2, (a.y + b.y) / 2 - 10 / k,
            "600 10px sans-serif", "#5160f0");
        }
      } else if (k > 1.3 && !faded) {
        ctx.fillStyle = "#8a93a6"; ctx.font = "9px sans-serif"; ctx.textAlign = "center";
        ctx.fillText(this.REL_LABEL[e.relation] || e.relation, (a.x + b.x) / 2, (a.y + b.y) / 2 - 3);
      }
      ctx.globalAlpha = 1;
    }
    // ---- 节点 ----
    for (const n of this.nodes) {
      if (typeOff.has(n.type)) continue;
      const size = n._r;
      const alpha = dim(n);
      if (alpha < 1) ctx.globalAlpha = alpha;
      const isFocus = n === focusNode;
      if (this.mode === "aggregate") {
        // 企业超节点：径向渐变（中心亮）+ 白描边 + 焦点光晕
        if (isFocus || n === this.selected) {
          ctx.shadowColor = "rgba(53,86,245,.55)"; ctx.shadowBlur = 22;
        }
        const g = ctx.createRadialGradient(n.x - size * .3, n.y - size * .35, size * .1, n.x, n.y, size);
        g.addColorStop(0, "#5d84ff");
        g.addColorStop(0.65, "#2f57ec");
        g.addColorStop(1, "#1e3fbe");
        ctx.beginPath(); ctx.arc(n.x, n.y, size, 0, Math.PI * 2);
        ctx.fillStyle = g; ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = "rgba(255,255,255,.92)";
        ctx.lineWidth = 2.2 / Math.max(k, 0.4);
        ctx.stroke();
        // 大节点内嵌事件数（白色，够大才画）
        if (size > 30 && k > 0.5) {
          ctx.fillStyle = "#fff"; ctx.font = `600 ${Math.min(15, size * 0.34)}px sans-serif`;
          ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ctx.fillText(String(n.n_events || ""), n.x, n.y);
          ctx.textBaseline = "alphabetic";
        }
        // 选中/悬停外环
        if (isFocus || n === this.selected) {
          ctx.strokeStyle = "#3556f5"; ctx.lineWidth = 2 / Math.max(k, 0.4);
          ctx.setLineDash([5, 4]);
          ctx.beginPath(); ctx.arc(n.x, n.y, size + 5, 0, Math.PI * 2); ctx.stroke();
          ctx.setLineDash([]);
        }
      } else {
        ctx.beginPath(); ctx.arc(n.x, n.y, size, 0, Math.PI * 2);
        ctx.fillStyle = this.COLORS[n.type] || "#4b5567";
        ctx.fill();
        if (isFocus || n === this.selected) {
          ctx.strokeStyle = "#101529"; ctx.lineWidth = 2.5;
          ctx.beginPath(); ctx.arc(n.x, n.y, size + 3.5, 0, Math.PI * 2); ctx.stroke();
        }
      }
      // ---- 标签 ----
      const label = n.label.length > 18 ? n.label.slice(0, 17) + "…" : n.label;
      if (this.mode === "aggregate") {
        if (k > 0.62 || isFocus || n === this.selected) {
          this.labelPill(ctx, `${label} · ${n.n_events}事件`, n.x, n.y + size + 15,
            "600 11.5px sans-serif", "#17213c");
          if (k > 0.85) {
            this.labelPill(ctx, `${n.tmin || "?"} ~ ${n.tmax || "?"}`, n.x, n.y + size + 34,
              "10px sans-serif", "#8a93a6");
          }
          // 类型分布徽标
          const types = Object.entries(n.types || {}).slice(0, 3);
          if (types.length && k > 0.85) {
            let tx = n.x - (types.length * 46) / 2 + 23;
            for (const [t] of types) {
              this.labelPill(ctx, (TYPE_LABEL[t] || t).slice(0, 3), tx, n.y - size - 16,
                "9px sans-serif", "#4b5567");
              tx += 46;
            }
          }
        }
      } else if (k > 0.9 || isFocus || n === this.selected) {
        this.labelPill(ctx, label, n.x, n.y + size + 14, "10.5px sans-serif", "#17213c");
      }
      ctx.globalAlpha = 1;
    }
    ctx.restore();
  },
  // 搜索定位：匹配节点 → 高亮 + 视野居中放大
  focusNode(n) {
    if (!n) return;
    const w = this.w || 900, h = this.h || 600;
    this.selected = n;
    const k = Math.max(this.tf.k, 1.15);
    this.tf = { k, x: w / 2 - n.x * k, y: h / 2 - n.y * k };
    this.draw();
    this.showDetail(n);
    const tip = $("#graphTip");
    tip.classList.remove("show");
  },
  // 搜索候选列表刷新（load 后调用）
  refreshSearchList() {
    const dl = $("#graphSearchList");
    if (!dl) return;
    dl.innerHTML = this.nodes.slice(0, 300).map(n =>
      `<option value="${esc(n.label)}">`).join("");
  },
  // 图例过滤与缩放控件绑定（一次性）
  bindToolbar() {
    if (this._toolbarBound) return;
    this._toolbarBound = true;
    this.typeOff = new Set();
    $("#zoomIn").addEventListener("click", () => this.zoomBy(1.25));
    $("#zoomOut").addEventListener("click", () => this.zoomBy(0.8));
    $("#zoomFit").addEventListener("click", () => { this.fitView(); this.draw(); });
    $("#graphSearch").addEventListener("change", e => {
      const q = e.target.value.trim();
      if (!q) return;
      const n = this.nodes.find(x => x.label === q) ||
                this.nodes.find(x => x.label.includes(q));
      if (n) this.focusNode(n);
      else toast("图中未找到: " + q, "err");
    });
    $("#graphLegend").addEventListener("click", e => {
      const chip = e.target.closest(".legend-chip");
      if (!chip) return;
      if (this.mode === "aggregate") { toast("聚合视图为单一实体类型，切到全图可按类型筛选"); return; }
      const t = chip.dataset.type;
      if (this.typeOff.has(t)) this.typeOff.delete(t);
      else this.typeOff.add(t);
      chip.classList.toggle("off", this.typeOff.has(t));
      this.draw();
    });
  },
  zoomBy(f) {
    const w = this.w || 900, h = this.h || 600;
    const k0 = this.tf.k;
    const k1 = Math.max(0.3, Math.min(4, k0 * f));
    this.tf.x = w / 2 - (w / 2 - this.tf.x) * (k1 / k0);
    this.tf.y = h / 2 - (h / 2 - this.tf.y) * (k1 / k0);
    this.tf.k = k1;
    this.draw();
  },
  toWorld(px, py) {
    return { x: (px - this.tf.x) / this.tf.k, y: (py - this.tf.y) / this.tf.k };
  },
  hit(px, py) {
    const p = this.toWorld(px, py);
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const n = this.nodes[i];
      let size = n._r + 2;
      if ((n.x - p.x) ** 2 + (n.y - p.y) ** 2 < size * size) return n;
    }
    return null;
  },
  bindEvents() {
    const c = this.canvas;
    let moved = false;
    c.addEventListener("mousedown", e => {
      const r = c.getBoundingClientRect();
      const n = this.hit(e.clientX - r.left, e.clientY - r.top);
      moved = false;
      if (n) { this.dragging = n; } else { this.panning = true; }
      this.lastPos = { x: e.clientX, y: e.clientY };
    });
    window.addEventListener("mousemove", e => {
      const tip = $("#graphTip");
      if (!this.dragging && !this.panning) {
        // hover 检测：非拖拽状态悬停节点 → 高亮 + tooltip
        const r = c.getBoundingClientRect();
        const inCanvas = e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
        const n = inCanvas ? this.hit(e.clientX - r.left, e.clientY - r.top) : null;
        if (n !== this.hover) {
          this.hover = n;
          this.draw();
        }
        if (n) {
          const sub = this.mode === "aggregate"
            ? `<div class="sub">${n.n_events} 事件 · ${n.tmin || "?"} ~ ${n.tmax || "?"} · 点击下钻</div>`
            : `<div class="sub">${{ entity: "实体", event: "事件", process: "过程", assertion: "断言" }[n.type] || n.type}${n.state ? " · " + n.state : ""}</div>`;
          tip.innerHTML = `<b>${esc(n.label)}</b>${sub}`;
          tip.style.left = (e.clientX - r.left) + "px";
          tip.style.top = (e.clientY - r.top) + "px";
          tip.classList.add("show");
          c.style.cursor = "pointer";
        } else {
          tip.classList.remove("show");
          c.style.cursor = this.panning ? "grabbing" : "grab";
        }
        return;
      }
      moved = true;
      tip.classList.remove("show");
      if (this.dragging) {
        const r = c.getBoundingClientRect();
        const p = this.toWorld(e.clientX - r.left, e.clientY - r.top);
        this.dragging.x = p.x; this.dragging.y = p.y;
        this.draw();
      } else if (this.lastPos) {
        this.tf.x += e.clientX - this.lastPos.x;
        this.tf.y += e.clientY - this.lastPos.y;
        this.lastPos = { x: e.clientX, y: e.clientY };
        this.draw();
      }
    });
    window.addEventListener("mouseup", e => {
      if (this.dragging && !moved) this.select(this.dragging);
      else if (this.panning && !moved) { this.selected = null; this.showDetail(null); }
      if (this.dragging && moved) {
        // 拖拽结束：局部重排 + 视野重适配
        this.relax(80); this.fitView(); this.draw();
      }
      this.dragging = null; this.panning = false;
    });
    c.addEventListener("wheel", e => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      const k0 = this.tf.k;
      const k1 = Math.max(0.35, Math.min(4, k0 * (e.deltaY < 0 ? 1.12 : 0.89)));
      this.tf.x = mx - (mx - this.tf.x) * (k1 / k0);
      this.tf.y = my - (my - this.tf.y) * (k1 / k0);
      this.tf.k = k1;
      this.draw();
    }, { passive: false });
    c.addEventListener("dblclick", () => { this.fitView(); this.draw(); });
  },
  select(n) {
    if (this.mode === "aggregate" && n.type === "entity") {
      // 企业聚合视图点击 → 切换到该企业 focus 下钻
      $("#graphMode").value = "full";
      // 预载下拉只有前 400 个实体，命中不了就动态补一个 option，保证 focus 不丢
      let opt = [...$("#graphFocus").options].find(o => o.value === n.raw_id);
      if (!opt) {
        opt = document.createElement("option");
        opt.value = n.raw_id; opt.textContent = n.label;
        $("#graphFocus").appendChild(opt);
      }
      $("#graphFocus").value = n.raw_id;
      this.load();
      return;
    }
    this.selected = n; this.showDetail(n);
  },
  async showDetail(n) {
    const pane = $("#graphDetail");
    if (!n) { pane.innerHTML = `<div class="empty"><div class="icon">⬡</div>点击图谱节点<br>查看详情与操作</div>`; return; }
    pane.innerHTML = `<div class="empty">加载中…</div>`;
    try {
      if (n.type === "entity") {
        const d = await api("/api/entities/" + n.raw_id);
        const aliases = (d.aliases || []).map(a => esc(a.alias)).join("、") || "—";
        const events = (d.events || []).map(e =>
          `<div style="padding:4px 0"><span class="badge ${e.state === "resolved" ? "green" : "orange"}">${TYPE_LABEL[e.event_type] || e.event_type}</span>
           ${esc(short(e.summary, 30))}</div>`).join("") || "<div class='tag'>无</div>";
        const factsHtml = (d.facts || []).filter(f => f.selections && f.selections.length).slice(0, 6).map(f => {
          const sel = f.selections[0];
          return `<div style="padding:4px 0;font-size:12.5px">
            <span class="mono" style="color:var(--ink-3)">${esc(f.slot_key.split(":")[1] || "")}</span>
            = <b class="mono">${esc(sel.value ? sel.value.value : "?")}</b>
            ${sel.disposition === "conflicted" ? '<span class="badge red">争议</span>' : ""}</div>`;
        }).join("") || "<div class='tag'>无当前事实</div>";
        pane.innerHTML = `
          <h3 class="sec">实体 · ${esc(d.canonical_name)}</h3>
          <div class="dl">
            <span class="k">类型</span><span>${esc(d.type)}</span>
            <span class="k">别名</span><span>${aliases}</span>
            <span class="k">ID</span><span class="mono" style="font-size:11px">${esc(d.entity_id)}</span>
          </div>
          <h3 class="sec mt">参与事件（${(d.events || []).length}）</h3>${events}
          <h3 class="sec mt">当前事实</h3>${factsHtml}
          <div class="row mt">
            <button class="btn ghost sm" onclick="ui.editEntity('${esc(d.entity_id)}')">编辑</button>
            <button class="btn ghost sm" style="color:var(--err)" onclick="ui.deleteEntity('${esc(d.entity_id)}')">删除</button>
          </div>`;
      } else if (n.type === "event") {
        const d = await api("/api/events/cluster/" + n.raw_id);
        const mem = (d.members || []).map(m => `<div style="padding:5px 0;border-bottom:1px solid #f0f2f8">
          <div>${esc(short(m.frame_text, 46))}</div>
          <div class="meta" style="font-size:11.5px;color:var(--ink-3)">${esc(m.source_name || "")}
          ${m.doc_title ? " · " + esc(short(m.doc_title, 20)) : ""}
          ${m.url ? ` · <a href="${esc(m.url)}" target="_blank" style="color:var(--accent)">原文</a>` : ""}</div>
          ${(m.evidence || []).slice(0, 1).map(e => `<div style="font-size:12px;color:var(--ink-2);
            background:#f8f9fd;border-radius:8px;padding:6px 8px;margin-top:4px">“${esc(short(e.quote, 64))}”</div>`).join("")}
        </div>`).join("") || "<div class='tag'>无成员</div>";
        pane.innerHTML = `
          <h3 class="sec">事件 · ${TYPE_LABEL[d.event_type] || d.event_type}
            <span class="badge ${d.state === "resolved" ? "green" : "orange"}">${esc(d.state)}</span>
            <span class="badge gray">v${d.version}</span></h3>
          <div class="dl">
            <span class="k">时间</span><span>${esc((d.event_time_lower || "?").slice(0, 10))} ~ ${esc((d.event_time_upper || "?").slice(0, 10))}</span>
            <span class="k">成员</span><span>${(d.members || []).length} 条提及</span>
          </div>
          <div style="font-size:12.5px;margin-top:8px;color:var(--ink-2)">${esc(d.summary)}</div>
          <h3 class="sec mt">成员证据</h3><div style="max-height:300px;overflow-y:auto">${mem}</div>
          <div class="row mt">
            <button class="btn ghost sm" onclick="ui.editEvent('${esc(d.cluster_id)}',${d.version})">编辑</button>
            <button class="btn ghost sm" onclick="ui.clusterOps('split','${esc(d.cluster_id)}',${d.version})">拆分</button>
            <button class="btn ghost sm" onclick="ui.clusterOps('merge','${esc(d.cluster_id)}',${d.version})">并入他簇</button>
            <button class="btn ghost sm" style="color:var(--err)" onclick="ui.deleteEvent('${esc(d.cluster_id)}',${d.version})">删除</button>
          </div>`;
      } else if (n.type === "process") {
        const { timeline: evts } = await api(`/api/timeline?process=${encodeURIComponent(n.raw_id)}`);
        pane.innerHTML = `<h3 class="sec">过程 · ${esc(n.label)}</h3>
          <div class="tl">${evts.map(e => `<div class="tl-item">
            <div><span class="badge blue">${TYPE_LABEL[e.event_type] || e.event_type}</span>
            <span style="font-size:12px;color:var(--ink-3)">${esc((e.event_time_lower || "?").slice(0, 10))}</span></div>
            <div style="font-size:12.5px;margin-top:2px">${esc(short(e.summary, 44))}</div>
          </div>`).join("") || "<div class='tag'>空</div>"}</div>`;
      } else if (n.type === "assertion") {
        pane.innerHTML = `<h3 class="sec">断言</h3>
          <div class="dl"><span class="k">槽位</span><span class="slot-path">${esc(n.label)}</span></div>
          <div class="row mt"><button class="btn ghost sm" onclick="ui.clusterOps('retract','${esc(n.raw_id)}')">撤回断言</button></div>`;
      }
    } catch (e) { pane.innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`; }
  },
};
$("#graphRefresh").addEventListener("click", () => graph.load());
$("#graphFocus").addEventListener("change", () => graph.load());
$("#graphMode").addEventListener("change", e => {
  $("#graphFocus").disabled = e.target.value === "aggregate";
  graph.load();
});
$("#graphAddNode").addEventListener("click", () => ui.createEntity());
$("#graphAddEvent").addEventListener("click", () => ui.createEvent());
$("#graphAddEdge").addEventListener("click", () => ui.createEdge());


/* ---------------- 实体演化史 ---------------- */
const entityHistory = {
  byLabel: new Map(),   // 显示名 → entity_id（datalist 搜索式选择）
  async loadOptions() {
    if (this._loaded) return;
    try {
      const g = await api("/api/graph?mode=aggregate");
      const dl = $("#histEntityList");
      const rows = g.nodes.filter(n => (n.n_events || 0) >= 3)
        .sort((a, b) => (b.n_events || 0) - (a.n_events || 0));
      this.byLabel = new Map(rows.map(n => [`${n.label}（${n.n_events}事件）`, n.raw_id]));
      // 追加同名简写映射（输入"小米集团"也能匹配）
      for (const n of rows) if (!this.byLabel.has(n.label)) this.byLabel.set(n.label, n.raw_id);
      dl.innerHTML = rows.map(n => `<option value="${esc(n.label)}（${n.n_events}事件）">`).join("");
      this._loaded = true;
    } catch (_) { /* 忽略 */ }
  },
  resolveId() {
    const v = $("#histEntity").value.trim();
    if (!v) return null;
    if (this.byLabel.has(v)) return this.byLabel.get(v);
    // 容错：去掉尾部（N事件）再试
    const bare = v.replace(/（\d+事件）$/, "");
    if (this.byLabel.has(bare)) return this.byLabel.get(bare);
    for (const [k, id] of this.byLabel) if (k.startsWith(bare)) return id;
    return null;
  },
  async load() {
    const eid = this.resolveId();
    if (!eid) { toast("请先从列表选择一个实体", "err"); return; }
    const box = $("#histBox");
    box.innerHTML = `<div class="empty">加载中…</div>`;
    try {
      const d = await api("/api/entity-history/" + encodeURIComponent(eid));
      $("#histSummary").textContent =
        `${d.entity.canonical_name}：${(d.events || []).length} 个事件，${(d.facts || []).length} 个事实槽位`;
      // 事件时间线
      const evs = (d.events || []).slice().sort((a, b) =>
        (a.event_time_lower || "").localeCompare(b.event_time_lower || ""));
      let html = `<div class="grid cols-2"><div><h3 class="sec" style="font-size:13px">事件时间线</h3><div class="tl">` +
        evs.map(e => `<div class="tl-item">
          <div><span class="badge blue">${TYPE_LABEL[e.event_type] || e.event_type}</span>
            <span style="font-size:12px;color:var(--ink-3)">${esc((e.event_time_lower || "?").slice(0, 10))}</span></div>
          <div style="font-size:12.5px">${esc(short(e.card_text || e.summary, 60))}</div>
        </div>`).join("") + `</div></div>`;
      // 槽位版本演化
      html += `<div><h3 class="sec" style="font-size:13px">槽位版本演化（双时间）</h3>` +
        (d.facts || []).map(f => `<div class="dec-detail" style="margin-bottom:8px">
          <div class="slot-path">${esc(f.slot_key)}</div>
          <table class="tbl" style="margin-top:6px"><thead><tr>
            <th>生效区间</th><th>值</th><th>状态</th></tr></thead><tbody>
            ${(f.versions || []).map(v => `<tr>
              <td class="mono" style="font-size:11px">${esc(v.valid_from)} ~ ${esc(v.valid_to)}</td>
              <td class="mono">${esc(String(v.value ?? "?"))}</td>
              <td>${v.disposition === "conflicted"
                  ? '<span class="badge red">争议</span>'
                  : v.disposition === "accepted" ? '<span class="badge green">生效</span>'
                  : '<span class="badge gray">未知</span>'}</td></tr>`).join("")}
          </tbody></table></div>`).join("") || "<div class='tag'>无</div>";
      html += `</div></div>`;
      box.innerHTML = html;
    } catch (e) {
      box.innerHTML = `<div class="empty" style="color:var(--err)">加载失败: ${esc(e.message)}</div>`;
    }
  },
};
$("#histLoad").addEventListener("click", () => entityHistory.load());
$("#histEntity").addEventListener("change", () => entityHistory.load());


/* ---------------- 事实与断言 ---------------- */
const facts = {
  async load() {
    entityHistory.loadOptions();
    // 兼容纯日期（10 字符）输入：补 T00:00；16 字符补秒
    const norm = v => !v ? "" : v.length === 10 ? v + "T00:00" : v;
    const va = norm($("#validAsOf").value), ka = norm($("#knownAsOf").value);
    const qs = new URLSearchParams();
    if (va) qs.set("valid_as_of", va.length === 16 ? va + ":00" : va);
    if (ka) qs.set("known_as_of", ka.length === 16 ? ka + ":00" : ka);
    try {
      const { facts: rows } = await api("/api/facts?" + qs.toString());
      const now = new Date();
      const mode = va && va < now.toISOString().slice(0, 16) ? (ka && ka <= va ? "当时已知口径" : "历史回看口径") : "当前口径";
      $("#factMode").textContent = `（${mode}）`;
      this._rows = rows.filter(f => f.selections && f.selections.length);
      this.render();
      this.applyFilter();
    } catch (e) { toast("事实查询失败: " + e.message, "err"); }
    this.loadTimeline();
  },
  render() {
    const rows = this._rows || [];
      $("#factList").innerHTML = rows.map(f => {
        const sel = f.selections[0];
        const rawVal = String(sel.value?.value ?? "");
        const unit = String(sel.value?.unit ?? "");
        const val = rawVal + (unit && !rawVal.trim().endsWith(unit.trim()) ? " " + esc(unit) : "");
        const disp = sel.disposition === "conflicted"
          ? `<span class="badge red">争议</span>` : sel.disposition === "accepted"
          ? `<span class="badge green">已选</span>` : `<span class="badge gray">未知</span>`;
        const conflicts = (f.conflicts || []).map(c =>
          `<div style="font-size:12px;color:var(--err);margin-top:4px">⚠ ${esc(c.value ? c.value.value : "?")}
          （${esc((c.recorded_at || "").slice(0, 10))}）</div>`).join("");
        const corrections = (f.corrections || []).map(c =>
          `<div style="font-size:12px;color:var(--violet);margin-top:4px">✎ 更正：原 ${esc(c.old ? c.old.value : "?")} → 现 ${esc(sel.value ? sel.value.value : "?")}</div>`).join("");
        return `<div class="card fact-card ${f.conflicts && f.conflicts.length ? "conflict" : ""}" data-search="${esc((f.subject || "") + " " + (f.slot_key || ""))}">
          <div class="row" style="justify-content:space-between">
            <b style="font-size:14px">${esc(f.subject || "")}</b>${disp}</div>
          <div class="slot-path mt" style="margin:6px 0">${esc(f.slot_key)}</div>
          <div class="row"><span class="val-box now">${val}</span>
            <span class="tag" style="font-size:11.5px;color:var(--ink-3)">
            有效自 ${esc((sel.valid_from || "?").slice(0, 10))}${sel.valid_to ? " 至 " + esc(sel.valid_to.slice(0, 10)) : " 起"}</span></div>
          ${conflicts}${corrections}
          ${f.empty_reason ? `<div class="tag" style="color:var(--warn)">${esc(f.empty_reason)}</div>` : ""}
        </div>`;
      }).join("") || `<div class="card"><div class="empty"><div class="icon">⏱</div>该时间口径下没有事实选择记录<br>试试调整时间点</div></div>`;
  },
  // 本地关键词过滤（主体/槽位），并更新计数
  applyFilter() {
    const q = ($("#factFilter").value || "").trim().toLowerCase();
    let shown = 0;
    document.querySelectorAll("#factList .fact-card").forEach(el => {
      const hit = !q || (el.dataset.search || "").toLowerCase().includes(q);
      el.style.display = hit ? "" : "none";
      if (hit) shown++;
    });
    const fc = $("#factCount");
    if (fc) fc.textContent = q ? `${shown} / ${(this._rows || []).length} 张` : `${(this._rows || []).length} 张`;
  },
  async loadTimeline() {
    try {
      const { timeline } = await api("/api/timeline?limit=80");
      $("#tlTable tbody").innerHTML = timeline.map(e => `<tr>
        <td class="mono" style="font-size:12px">${esc((e.event_time_lower || "未知").slice(0, 10))}</td>
        <td><span class="badge blue">${TYPE_LABEL[e.event_type] || e.event_type}</span></td>
        <td>${esc(short(e.summary, 52))}${e.process_title ? `<div class="tag" style="font-size:11px;color:var(--violet)">⑂ ${esc(e.process_title)}</div>` : ""}</td>
        <td class="mono">${e.members}</td>
        <td><span class="badge ${e.state === "resolved" ? "green" : "orange"}">${e.state}</span></td></tr>`).join("")
        || `<tr><td colspan="5"><div class="empty">暂无事件</div></td></tr>`;
    } catch (_) { /* 忽略 */ }
  },
};
$("#factQuery").addEventListener("click", () => facts.load());
$("#factNow").addEventListener("click", () => {
  $("#validAsOf").value = ""; $("#knownAsOf").value = ""; facts.load();
});
$("#factFilter").addEventListener("input", () => facts.applyFilter());

/* ---------------- 变化卡 ---------------- */
const changes = {
  async load() {
    try {
      const { changes: rows } = await api("/api/changes?limit=200");
      $("#changeList").innerHTML = rows.map(c => {
        const before = c.before_json ? (() => { try { return JSON.parse(c.before_json).value; } catch (_) { return "?"; } })() : null;
        const after = c.after_json ? (() => { try { return JSON.parse(c.after_json).value; } catch (_) { return "?"; } })() : null;
        return `<div class="card change-card mt" style="margin-top:0;margin-bottom:12px">
          <div class="row" style="justify-content:space-between">
            <span class="badge ${CHANGE_CLASS[c.change_kind] || "gray"}">${CHANGE_LABEL[c.change_kind] || c.change_kind}</span>
            <span class="tag" style="font-size:11.5px;color:var(--ink-3)">${fmtTime(c.created_at)}</span>
          </div>
          <div style="margin-top:8px"><b>${esc(c.subject || "")}</b>
            <span class="slot-path"> ${esc(c.slot_key.split(":").slice(1).join(":"))}</span></div>
          <div class="before-after">
            ${before != null ? `<span class="val-box old">${esc(before)}</span><span>→</span>` : ""}
            <span class="val-box now">${esc(after ?? "（撤回）")}</span>
          </div>
        </div>`;
      }).join("") || `<div class="empty"><div class="icon">🔔</div>暂无变化记录<br>运行流水线后，价格/状态等白名单谓词的真实变化会出现在这里</div>`;
    } catch (e) { toast("变化卡加载失败: " + e.message, "err"); }
  },
};

/* ---------------- 问答 ---------------- */
const qa = {
  busy: false, evidence: [],
  send() {
    const q = $("#qaInput").value.trim();
    if (!q || this.busy) return;
    $("#qaInput").value = "";
    this.busy = true;
    $("#qaSend").disabled = true;
    const msgs = $("#qaMsgs");
    $(".empty", msgs) && $(".empty", msgs).remove();
    const u = document.createElement("div");
    u.className = "msg user"; u.textContent = q;
    msgs.appendChild(u);
    const ai = document.createElement("div");
    ai.className = "msg ai waiting";
    ai.innerHTML = "正在检索证据…<span class='cursor-blink'></span>";
    msgs.appendChild(u);
    msgs.appendChild(ai);
    msgs.scrollTop = msgs.scrollHeight;
    const mdBox = document.createElement("div");
    this.stream(q, ai, mdBox);
  },
  async stream(q, aiEl, mdBox) {
    this.evidence = [];
    let answer = "";
    try {
      const resp = await fetch("/api/answer", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
          const line = frame.split("\n").find(l => l.startsWith("data:"));
          if (!line) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") continue;
          let ev; try { ev = JSON.parse(data); } catch (_) { continue; }
          if (ev.type === "evidence") {
            this.renderEvidence(ev.pack);
            aiEl.innerHTML = "证据就绪，生成回答中…<span class='cursor-blink'></span>";
          } else if (ev.type === "token") {
            answer += ev.text;
            aiEl.classList.remove("waiting");
            aiEl.innerHTML = md(answer) + '<span class="cursor-blink"></span>';
            const msgs = $("#qaMsgs");
            msgs.scrollTop = msgs.scrollHeight;
          } else if (ev.type === "done") {
            answer = ev.answer || answer;
            aiEl.innerHTML = md(answer);
          } else if (ev.type === "error") {
            aiEl.classList.remove("waiting");
            aiEl.innerHTML = `<span style="color:var(--err)">生成失败：${esc(ev.message)}</span>`;
          }
        }
      }
    } catch (e) {
      aiEl.classList.remove("waiting");
      aiEl.innerHTML = `<span style="color:var(--err)">请求失败：${esc(e.message)}</span>`;
    } finally {
      this.busy = false;
      $("#qaSend").disabled = false;
    }
  },
  renderEvidence(pack) {
    this.evidence = pack.evidence || [];
    const factsHtml = (pack.facts || []).map(f => {
      const sel = (f.selections || [])[0];
      return `<div class="ev-item"><b>结构化事实</b> · ${esc(f.subject || "")}
        <div class="mono" style="font-size:11px;color:var(--ink-3);margin-top:2px">${esc(f.slot_key)}</div>
        <div style="margin-top:4px">${esc(sel && sel.value ? sel.value.value : "?")}
          ${sel && sel.disposition === "conflicted" ? '<span class="badge red">争议</span>' : ""}</div>
        <div class="tag" style="font-size:11px;color:var(--ink-3)">valid=${esc((f.valid_as_of || "").slice(0, 16))} known=${esc((f.known_as_of || "").slice(0, 16))}</div></div>`;
    }).join("");
    const evHtml = this.evidence.map(e => `<div class="ev-item">
      <span class="no">${e.no}</span>${e.url ? `<a href="${esc(e.url)}" target="_blank">${esc(short(e.title, 34))}</a>` : esc(short(e.title, 34))}
      <div style="color:var(--ink-2);margin-top:4px;font-size:12px">${esc(short(e.snippet, 90))}</div>
      <div class="tag" style="font-size:11px;color:var(--ink-3)">${esc(e.source || "")} · ${esc((e.published_at || "").slice(0, 10))}</div></div>`).join("");
    const confHtml = (pack.conflicts || []).length
      ? `<div class="ev-item" style="border-color:#f5c6c8"><b style="color:var(--err)">⚠ 存在争议</b></div>` : "";
    const corrHtml = (pack.corrections || []).length
      ? `<div class="ev-item" style="border-color:#e2d5fb"><b style="color:var(--violet)">✎ 含更正记录</b></div>` : "";
    $("#qaEvidence").innerHTML = factsHtml + evHtml + confHtml + corrHtml ||
      `<div class="empty">无证据</div>`;
    if (!factsHtml && !evHtml) $("#qaEvidence").innerHTML =
      `<div class="empty"><div class="icon">🗂</div>本次未命中证据</div>`;
  },
};
$("#qaSend").addEventListener("click", () => qa.send());
$("#qaInput").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); qa.send(); }
});

/* ---------------- 极简 Markdown（含引用角标） ---------------- */
function md(src) {
  let s = esc(src);
  s = s.replace(/```([\s\S]*?)```/g, (_, c) => `<pre style="background:#eceff7;border-radius:8px;padding:8px 10px;overflow-x:auto;font-size:12px;margin:.4em 0">${c.trim()}</pre>`);
  s = s.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  s = s.replace(/^####?\s*(.+)$/gm, "<h4>$1</h4>");
  s = s.replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>");
  s = s.replace(/^\s*[-*]\s+(.+)$/gm, "<li>$1</li>");
  s = s.replace(/(<li>[\s\S]*?<\/li>)(?!\s*<li>)/g, m => `<ul>${m}</ul>`);
  s = s.replace(/\[(\d{1,2})\]/g, '<span class="cite" data-cite="$1">$1</span>');
  s = s.split(/\n{2,}/).map(p => /^<(ul|pre|h4)/.test(p.trim()) ? p : `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
  return `<div class="md">${s}</div>`;
}
document.addEventListener("click", e => {
  const c = e.target.closest(".cite");
  if (!c) return;
  const ev = qa.evidence.find(x => x.no === +c.dataset.cite);
  if (ev) openModal(`证据 [${ev.no}] · ${ev.title || ""}`, `
    <div class="dl">
      <span class="k">来源</span><span>${esc(ev.source || "")}</span>
      <span class="k">发布时间</span><span>${esc(ev.published_at || "未知")}</span>
      <span class="k">链接</span><span>${ev.url ? `<a href="${esc(ev.url)}" target="_blank">${esc(ev.url)}</a>` : "无"}</span>
    </div>
    <div class="dec-detail mt">${esc(ev.snippet || "")}</div>`);
});

/* ---------------- 弹窗与 CRUD 表单 ---------------- */
function openModal(title, bodyHTML) {
  $("#modalTitle").textContent = title;
  $("#modalBody").innerHTML = bodyHTML;
  $("#mask").classList.add("open");
}
$("#modalClose").addEventListener("click", () => $("#mask").classList.remove("open"));
$("#mask").addEventListener("click", e => { if (e.target === $("#mask")) $("#mask").classList.remove("open"); });

const ui = {
  form(html) { return html; },
  async createEntity() {
    openModal("新建实体", `
      <div class="form-row"><label>实体名 *</label><input class="input" id="f_name"></div>
      <div class="form-cols">
        <div class="form-row"><label>类型</label><select class="input" id="f_type">
          ${["company", "brand", "product", "person", "other"].map(t => `<option>${t}</option>`).join("")}
        </select></div>
        <div class="form-row"><label>别名（逗号分隔）</label><input class="input" id="f_aliases"></div>
      </div>
      <div class="form-row"><label>备注</label><input class="input" id="f_note"></div>
      <div class="row" style="justify-content:flex-end"><button class="btn" id="f_ok">创建</button></div>`);
    $("#f_ok").addEventListener("click", async () => {
      try {
        await api("/api/entities", { method: "POST", body: {
          name: $("#f_name").value, type: $("#f_type").value,
          aliases: $("#f_aliases").value.split(/[,，]/).map(s => s.trim()).filter(Boolean),
          note: $("#f_note").value || null } });
        $("#mask").classList.remove("open");
        toast("实体已创建", "ok"); graph._focusLoaded = false; graph.load();
      } catch (e) { toast(e.message, "err"); }
    });
  },
  async editEntity(id) {
    try {
      const d = await api("/api/entities/" + id);
      openModal(`编辑实体 · ${d.canonical_name}`, `
        <div class="form-row"><label>规范名</label><input class="input" id="f_name" value="${esc(d.canonical_name)}"></div>
        <div class="form-row"><label>追加别名（逗号分隔）</label><input class="input" id="f_aliases" placeholder="如：小米集团, Xiaomi"></div>
        <div class="form-row"><label>乐观锁 · updated_at</label><input class="input" id="f_ver" value="${esc(d.updated_at || "")}"></div>
        <div class="row" style="justify-content:flex-end"><button class="btn" id="f_ok">保存</button></div>`);
      $("#f_ok").addEventListener("click", async () => {
        try {
          await api("/api/entities/" + id, { method: "PATCH", body: {
            name: $("#f_name").value, expected_updated: $("#f_ver").value || null,
            add_aliases: $("#f_aliases").value.split(/[,，]/).map(s => s.trim()).filter(Boolean) } });
          $("#mask").classList.remove("open");
          toast("实体已更新", "ok"); graph.load(); graph.showDetail(graph.selected);
        } catch (e) { toast(e.status === 409 ? "版本冲突（409）: " + e.message : e.message, "err"); }
      });
    } catch (e) { toast(e.message, "err"); }
  },
  async deleteEntity(id) {
    openModal("删除实体（软删除）", `
      <p style="font-size:13px;color:var(--ink-2)">将软删除该实体并移除其关系边；仍参与事件的实体会被拒绝。</p>
      <div class="row" style="justify-content:flex-end;margin-top:14px">
        <button class="btn ghost" onclick="document.getElementById('mask').classList.remove('open')">取消</button>
        <button class="btn danger" id="f_ok">确认删除</button></div>`);
    $("#f_ok").addEventListener("click", async () => {
      try {
        await api("/api/entities/" + id, { method: "DELETE", body: {} });
        $("#mask").classList.remove("open");
        toast("实体已删除", "ok"); graph.load(); graph.showDetail(null);
      } catch (e) { toast(e.message, "err"); }
    });
  },
  async createEvent() {
    let entOpts = "";
    try { const { nodes } = await api("/api/nodes/entity?limit=300");
      entOpts = nodes.map(n => `<option value="${esc(n.id)}">${esc(n.title)}</option>`).join(""); } catch (_) { /* 空 */ }
    openModal("新建人工事件", `
      <div class="form-row"><label>标题 *</label><input class="input" id="f_title"></div>
      <div class="form-cols">
        <div class="form-row"><label>事件类型</label><select class="input" id="f_type">
          ${Object.entries(TYPE_LABEL).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}
        </select></div>
        <div class="form-row"><label>发生时间（YYYY-MM-DD）</label><input class="input" id="f_time" placeholder="2026-09-17"></div>
      </div>
      <div class="form-row"><label>关联实体（可多选）</label>
        <select class="input" id="f_ents" multiple size="4" style="width:100%">${entOpts}</select></div>
      <div class="row" style="justify-content:flex-end"><button class="btn" id="f_ok">创建</button></div>`);
    $("#f_ok").addEventListener("click", async () => {
      const t = $("#f_time").value;
      try {
        await api("/api/manual-events", { method: "POST", body: {
          title: $("#f_title").value, event_type: $("#f_type").value,
          event_time_lower: t ? t + "T00:00:00+08:00" : null,
          event_time_upper: t ? t + "T00:00:00+08:00" : null,
          entity_ids: [...$("#f_ents").selectedOptions].map(o => o.value) } });
        $("#mask").classList.remove("open");
        toast("事件已创建（人工，v1）", "ok"); graph.load();
      } catch (e) { toast(e.message, "err"); }
    });
  },
  async editEvent(id, version) {
    try {
      const d = await api("/api/events/cluster/" + id);
      openModal(`编辑事件（当前 v${version}）`, `
        <div class="form-row"><label>摘要</label><input class="input" id="f_summary" value="${esc(d.summary)}"></div>
        <div class="form-cols">
          <div class="form-row"><label>事件类型</label><select class="input" id="f_type">
            ${Object.entries(TYPE_LABEL).map(([k, v]) =>
              `<option value="${k}" ${k === d.event_type ? "selected" : ""}>${v}</option>`).join("")}
          </select></div>
          <div class="form-row"><label>状态</label><select class="input" id="f_state">
            ${["provisional", "resolved", "ambiguous"].map(s =>
              `<option ${s === d.state ? "selected" : ""}>${s}</option>`).join("")}
          </select></div>
        </div>
        <div class="form-row"><label>expected_version（乐观锁）</label>
          <input class="input mono" id="f_ver" value="${version}" readonly></div>
        <div class="row" style="justify-content:flex-end"><button class="btn" id="f_ok">保存（版本 +1）</button></div>`);
      $("#f_ok").addEventListener("click", async () => {
        try {
          await api("/api/manual-events/" + id, { method: "PATCH", body: {
            expected_version: +$("#f_ver").value, summary: $("#f_summary").value,
            event_type: $("#f_type").value, state: $("#f_state").value } });
          $("#mask").classList.remove("open");
          toast("事件已更新", "ok"); graph.load(); graph.showDetail(graph.selected);
        } catch (e) { toast(e.status === 409 ? "版本冲突（409）: " + e.message : e.message, "err"); }
      });
    } catch (e) { toast(e.message, "err"); }
  },
  async deleteEvent(id, version) {
    openModal("删除事件（软删除）", `
      <p style="font-size:13px;color:var(--ink-2)">将关闭成员归属、软删除事件并失效其关系边；仍含成员时需强制。</p>
      <div class="row" style="margin-top:10px"><label style="font-size:12.5px">
        <input type="checkbox" id="f_force"> 强制（连成员一起关闭）</label></div>
      <div class="row" style="justify-content:flex-end;margin-top:14px">
        <button class="btn ghost" onclick="document.getElementById('mask').classList.remove('open')">取消</button>
        <button class="btn danger" id="f_ok">确认删除</button></div>`);
    $("#f_ok").addEventListener("click", async () => {
      try {
        await api("/api/manual-events/" + id, { method: "DELETE", body: {
          expected_version: version, force: $("#f_force").checked } });
        $("#mask").classList.remove("open");
        toast("事件已删除", "ok"); graph.load(); graph.showDetail(null);
      } catch (e) { toast(e.status === 409 ? "版本冲突（409）: " + e.message : e.message, "err"); }
    });
  },
  async createEdge() {
    const nodeSel = async (id) => {
      let opts = "";
      try {
        const ents = await api("/api/nodes/entity?limit=200");
        const evs = await api("/api/nodes/event?limit=200");
        opts = `<optgroup label="实体">${ents.nodes.map(n =>
          `<option value="entity:${esc(n.id)}">${esc(n.title)}</option>`).join("")}</optgroup>
          <optgroup label="事件">${evs.nodes.map(n =>
          `<option value="event:${esc(n.id)}">${esc(short(n.title, 26))}</option>`).join("")}</optgroup>`;
      } catch (_) { /* 空 */ }
      return opts;
    };
    const opts = await nodeSel();
    openModal("新建语义关系边", `
      <div class="form-cols">
        <div class="form-row"><label>起点</label><select class="input" id="f_from">${opts}</select></div>
        <div class="form-row"><label>终点</label><select class="input" id="f_to">${opts}</select></div>
      </div>
      <div class="form-row"><label>关系类型</label><select class="input" id="f_rel">
        ${["participates_in", "part_of_process", "precedes", "related_to", "supports", "contradicts"]
          .map(r => `<option>${r}</option>`).join("")}</select></div>
      <div class="form-row"><label>备注</label><input class="input" id="f_note"></div>
      <div class="row" style="justify-content:flex-end"><button class="btn" id="f_ok">创建</button></div>`);
    $("#f_ok").addEventListener("click", async () => {
      const [ft, fid] = $("#f_from").value.split(":");
      const [tt, tid] = $("#f_to").value.split(":");
      try {
        await api("/api/edges", { method: "POST", body: {
          from_type: ft, from_id: fid, to_type: tt, to_id: tid,
          relation: $("#f_rel").value, note: $("#f_note").value || null } });
        $("#mask").classList.remove("open");
        toast("边已创建", "ok"); graph.load();
      } catch (e) { toast(e.message, "err"); }
    });
  },
  async clusterOps(kind, clusterId, version) {
    if (kind === "retract") {
      openModal("撤回断言", `
        <div class="form-row"><label>撤回依据 *</label><input class="input" id="f_basis" placeholder="如：来源已撤回 / 人工核实错误"></div>
        <div class="row" style="justify-content:flex-end"><button class="btn danger" id="f_ok">确认撤回</button></div>`);
      $("#f_ok").addEventListener("click", async () => {
        try {
          await api("/api/cluster-ops", { method: "POST", body: {
            op: "retract_assertion", assertion_id: clusterId, basis: $("#f_basis").value } });
          $("#mask").classList.remove("open");
          toast("断言已撤回", "ok"); graph.load();
        } catch (e) { toast(e.message, "err"); }
      });
      return;
    }
    if (kind === "split") {
      try {
        const d = await api("/api/events/cluster/" + clusterId);
        openModal(`拆分簇（当前 v${version}）`, `
          <p style="font-size:12.5px;color:var(--ink-2)">选择要拆出去的成员（其余留在原簇）：</p>
          ${d.members.map(m => `<label style="display:block;padding:5px 0;font-size:13px">
            <input type="checkbox" class="f_mem" value="${esc(m.mention_id)}">
            ${esc(short(m.frame_text, 50))}</label>`).join("")}
          <div class="form-row mt"><label>理由 *</label><input class="input" id="f_reason"></div>
          <div class="row" style="justify-content:flex-end"><button class="btn" id="f_ok">执行拆分</button></div>`);
        $("#f_ok").addEventListener("click", async () => {
          const ids = $$(".f_mem").filter(c => c.checked).map(c => c.value);
          try {
            const out = await api("/api/cluster-ops", { method: "POST", body: {
              op: "split_cluster", cluster_id: clusterId, mention_ids: ids,
              reason: $("#f_reason").value || "人工拆分", expected_version: version } });
            $("#mask").classList.remove("open");
            toast(`已拆出新簇 ${out.new_cluster.slice(0, 8)}…`, "ok"); graph.load();
          } catch (e) { toast(e.status === 409 ? "版本冲突（409）" : e.message, "err"); }
        });
      } catch (e) { toast(e.message, "err"); }
      return;
    }
    if (kind === "merge") {
      let opts = "";
      try {
        const { nodes } = await api("/api/nodes/event?limit=300");
        opts = nodes.filter(n => n.id !== clusterId).map(n =>
          `<option value="${esc(n.id)}">${esc(short(n.title, 40))}</option>`).join("");
      } catch (_) { /* 空 */ }
      openModal(`合并簇（源：${clusterId.slice(0, 8)}…）`, `
        <div class="form-row"><label>并入目标簇 *</label><select class="input" id="f_target">${opts}</select></div>
        <div class="form-row"><label>理由 *</label><input class="input" id="f_reason"></div>
        <div class="form-row"><label>目标簇 expected_version *</label>
          <input class="input mono" id="f_ver" type="number" placeholder="目标簇当前版本号"></div>
        <div class="row" style="justify-content:flex-end"><button class="btn" id="f_ok">执行合并</button></div>`);
      $("#f_ok").addEventListener("click", async () => {
        try {
          const out = await api("/api/cluster-ops", { method: "POST", body: {
            op: "merge_clusters", source_ids: [clusterId],
            target_cluster_id: $("#f_target").value,
            reason: $("#f_reason").value || "人工合簇",
            expected_version: +$("#f_ver").value } });
          $("#mask").classList.remove("open");
          toast(`已合并 ${out.moved.length} 个成员`, "ok"); graph.load();
        } catch (e) { toast(e.status === 409 ? "版本冲突（409）" : e.message, "err"); }
      });
    }
  },
};

/* ---------------- 启动 ---------------- */
feed.connect();
overview.refresh();
overview.renderStagesFromDB();
loadMiniFeed();
setInterval(() => { loadMiniFeed(); overview.refreshSoon(); }, 15000);
