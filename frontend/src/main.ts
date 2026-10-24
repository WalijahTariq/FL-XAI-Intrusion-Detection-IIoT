import "./style.css";
import type { FeatureImportance, MetricName, MetricRow, Results } from "./types";

// Hyperparameters published in the paper (Table I). Used only to flag values
// that the current code has changed, so edits show up clearly on the page.
const PAPER_TABLE_I: Record<"xgboost_params" | "lightgbm_params", Record<string, number>> = {
  xgboost_params: {
    n_estimators: 215, max_depth: 15, learning_rate: 0.101, subsample: 0.884,
    colsample_bytree: 0.809, gamma: 0.187, min_child_weight: 3,
  },
  lightgbm_params: {
    n_estimators: 291, max_depth: -1, learning_rate: 0.077, subsample: 0.622,
    colsample_bytree: 0.898, num_leaves: 74, min_child_samples: 24, reg_alpha: 0.027, reg_lambda: 0.013,
  },
};

const METRICS: { key: MetricName; label: string }[] = [
  { key: "accuracy", label: "Accuracy" },
  { key: "balanced_accuracy", label: "Balanced acc." },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1" },
];

// ---------------------------------------------------------------- helpers --
function esc(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

const pct = (v: number | null | undefined, digits = 2): string =>
  v == null ? "—" : `${(v * 100).toFixed(digits)}%`;
const num = (v: number | null | undefined, digits = 4): string =>
  v == null ? "—" : v.toFixed(digits);
const int = (v: number | null | undefined): string =>
  v == null ? "—" : Math.round(v).toLocaleString("en-US");

function signedPts(diff: number): string {
  const pts = diff * 100;
  return `${pts >= 0 ? "+" : "−"}${Math.abs(pts).toFixed(2)} pts`;
}

function section(id: string, title: string, intro: string, body: string): string {
  return `<section class="card" id="${id}">
    <h2>${esc(title)}</h2>
    ${intro ? `<p class="intro">${intro}</p>` : ""}
    ${body}
  </section>`;
}

function table(head: string[], rows: string[][], opts: { numericFrom?: number; rowClass?: (i: number) => string } = {}): string {
  const from = opts.numericFrom ?? 1;
  const th = head.map((h, i) => `<th class="${i >= from ? "num" : ""}">${esc(h)}</th>`).join("");
  const tr = rows
    .map((r, ri) => `<tr class="${opts.rowClass?.(ri) ?? ""}">${r
      .map((c, i) => `<td class="${i >= from ? "num" : ""}">${c}</td>`)
      .join("")}</tr>`)
    .join("");
  return `<div class="table-wrap"><table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`;
}

// ---------------------------------------------------------------- sections --
function renderBuild(r: Results): void {
  const b = r.build;
  const when = new Date(b.generated_at).toLocaleString();
  const source = b.source === "github-actions" ? "GitHub Actions" : "local run";
  const message = (b.commit_message ?? "").split("\n")[0];

  const first: string[] = [
    `<span class="pill ${b.source === "github-actions" ? "pill-ci" : "pill-local"}">Built by ${esc(source)}</span>`,
  ];
  if (b.commit_sha) {
    const sha = esc(b.commit_sha.slice(0, 7));
    first.push(`<span>Commit <code>${b.commit_url ? `<a href="${esc(b.commit_url)}">${sha}</a>` : sha}</code></span>`);
    if (message) first.push(`<span class="commit-msg">“${esc(message)}”</span>`);
  } else {
    first.push(`<span>No git commit information (not run inside a git repository)</span>`);
  }

  const second: string[] = [];
  const who = [b.commit_author ? esc(b.commit_author) : "", b.branch ? `branch <code>${esc(b.branch)}</code>` : ""]
    .filter(Boolean)
    .join(" · ");
  if (who) second.push(`<span>${who}</span>`);
  second.push(`<span>Results generated ${esc(when)} in ${int(b.duration_seconds)} s</span>`);
  if (b.run_url) second.push(`<a href="${esc(b.run_url)}">View this pipeline run →</a>`);

  document.getElementById("paper-title")!.textContent = `${r.paper.title} (${r.paper.venue})`;
  document.getElementById("build")!.innerHTML =
    `<div class="build-row">${first.join("")}</div><div class="build-row muted">${second.join("")}</div>`;
}

function renderHeadline(r: Results): string {
  const final = r.metrics.find((m) => m.model === r.final_model);
  const paperAcc = r.paper.benchmark_targets.accuracy;
  const diff = final && paperAcc != null ? final.accuracy - paperAcc : null;
  const card = (label: string, value: string, note: string) =>
    `<div class="stat"><div class="stat-label">${esc(label)}</div><div class="stat-value">${value}</div><div class="stat-note">${note}</div></div>`;
  return `<div class="stats">
    ${card(`${r.final_model} accuracy`, pct(final?.accuracy), diff == null ? "" : `${signedPts(diff)} vs paper (${pct(paperAcc)})`)}
    ${card("Weighted F1", pct(final?.f1), `paper: ${pct(r.paper.benchmark_targets.f1)}`)}
    ${card("Dataset rows", int(r.dataset.raw_rows), `${r.dataset.raw_columns} columns · ${esc(r.dataset.file)}`)}
    ${card("Features · classes", `${r.dataset.n_features} · ${r.dataset.classes.length}`, `train ${int(r.dataset.train_rows)} / test ${int(r.dataset.test_rows)}`)}
  </div>`;
}

function renderModelChart(r: Results): string {
  // Dot plot on a zoomed axis: all models score within a few points of each
  // other, so a zero-based bar chart would hide every difference.
  const keys: MetricName[] = ["accuracy", "f1"];
  const paperAcc = r.paper.benchmark_targets.accuracy;
  const values = r.metrics.flatMap((m) => keys.map((k) => m[k]));
  if (paperAcc != null) values.push(paperAcc);
  const lo = Math.max(0, Math.floor((Math.min(...values) - 0.005) * 100) / 100);
  const hi = Math.min(1, Math.ceil((Math.max(...values) + 0.002) * 100) / 100);
  const span = hi - lo || 0.01;
  const pos = (v: number) => `${(((v - lo) / span) * 100).toFixed(2)}%`;
  const step = span > 0.2 ? 0.05 : span > 0.06 ? 0.02 : 0.01;
  const ticks: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(Number(t.toFixed(4)));

  const rows = r.metrics
    .map((m) => `<div class="dot-row">
      <div class="dot-label${m.model === r.final_model ? " is-final" : ""}">${esc(m.model)}</div>
      <div class="dot-track">
        ${paperAcc != null ? `<span class="paper-line" style="left:${pos(paperAcc)}"></span>` : ""}
        <span class="dot dot-f1" style="left:${pos(m.f1)}" title="F1 ${pct(m.f1)}"></span>
        <span class="dot dot-acc" style="left:${pos(m.accuracy)}" title="Accuracy ${pct(m.accuracy)}"></span>
      </div>
      <div class="dot-value">${pct(m.accuracy)}</div>
    </div>`)
    .join("");
  const axis = `<div class="dot-row axis"><div></div><div class="dot-track axis-track">${ticks
    .map((t) => `<span class="tick" style="left:${pos(t)}">${(t * 100).toFixed(0)}%</span>`)
    .join("")}</div><div></div></div>`;
  const legend = `<div class="legend">
    <span><i class="dot dot-acc static"></i>Accuracy</span>
    <span><i class="dot dot-f1 static"></i>Weighted F1</span>
    ${paperAcc != null ? `<span><i class="paper-key"></i>Paper accuracy (${pct(paperAcc)})</span>` : ""}
    <span class="muted">Axis starts at ${(lo * 100).toFixed(0)}%, not 0, so small gaps are visible.</span>
  </div>`;
  return `<div class="dot-plot">${rows}${axis}</div>${legend}`;
}

function renderMetricsTable(r: Results): string {
  const best: Partial<Record<MetricName, number>> = {};
  for (const { key } of METRICS) best[key] = Math.max(...r.metrics.map((m) => m[key]));
  const rows = r.metrics.map((m: MetricRow) => [
    esc(m.model) + (m.model === r.final_model ? ' <span class="tag">final</span>' : ""),
    ...METRICS.map(({ key }) => `<span class="${m[key] === best[key] ? "best" : ""}">${num(m[key])}</span>`),
  ]);
  return table(["Model", ...METRICS.map((m) => m.label)], rows);
}

function renderPaperComparison(r: Results): string {
  const rows = r.paper_comparison.map((c) => [
    esc(METRICS.find((m) => m.key === c.metric)?.label ?? c.metric),
    num(c.paper),
    num(c.reproduction),
    `<span class="${c.difference >= 0 ? "up" : "down"}">${signedPts(c.difference)}</span>`,
  ]);
  return table(["Metric", "Paper", "This run", "Difference"], rows);
}

function renderPerClass(r: Results): string {
  const rows = r.per_class.map((c) => [esc(c.class), num(c.precision), num(c.recall), num(c.f1), int(c.support)]);
  return table(["Class", "Precision", "Recall", "F1", "Test rows"], rows);
}

function renderConfusion(r: Results): string {
  const { labels, matrix } = r.confusion_matrix;
  const head = `<tr><th class="corner">True ↓ / Predicted →</th>${labels
    .map((l) => `<th class="cm-col"><span>${esc(l)}</span></th>`)
    .join("")}</tr>`;
  const body = matrix
    .map((row, i) => {
      const total = row.reduce((a, b) => a + b, 0) || 1;
      const cells = row
        .map((v, j) => {
          const share = v / total;
          const strength = Math.round(Math.sqrt(share) * 85);
          const strong = share > 0.45 ? " strong" : "";
          const cls = i === j ? "diag" : "off";
          return `<td class="cm-cell ${cls}${strong}" style="--s:${strength}%" title="${esc(labels[i])} → ${esc(labels[j])}: ${int(v)} (${pct(share, 1)} of row)">${v === 0 ? "" : int(v)}</td>`;
        })
        .join("");
      return `<tr><th class="cm-row">${esc(labels[i])}</th>${cells}</tr>`;
    })
    .join("");
  return `<div class="table-wrap"><table class="cm"><thead>${head}</thead><tbody>${body}</tbody></table></div>
    <p class="hint">Diagonal cells are correct predictions; shading shows each cell’s share of its row. Hover a cell for details.</p>`;
}

function bars(items: FeatureImportance[], valueKey: "importance" | "mean_abs_shap", limit = 15): string {
  const list = items.slice(0, limit);
  const max = Math.max(...list.map((f) => f[valueKey] ?? 0)) || 1;
  return `<div class="bars">${list
    .map((f) => {
      const v = f[valueKey] ?? 0;
      return `<div class="bar-row"><span class="bar-label">${esc(f.feature)}</span>
        <span class="bar-track"><span class="bar" style="width:${((v / max) * 100).toFixed(1)}%"></span></span>
        <span class="bar-value">${v.toFixed(4)}</span></div>`;
    })
    .join("")}</div>`;
}

function renderImportance(r: Results): string {
  const shapModels = Object.keys(r.shap_top_features);
  const shapBlock = shapModels.length
    ? `<div class="toggle" role="tablist">${shapModels
        .map((m, i) => `<button type="button" role="tab" data-shap="${esc(m)}" aria-selected="${i === 0}">${esc(m)}</button>`)
        .join("")}</div>
       <div id="shap-bars">${bars(r.shap_top_features[shapModels[0]], "mean_abs_shap")}</div>`
    : `<p class="muted">SHAP was skipped in this run.</p>`;
  return `<div class="two-col">
    <div><h3>Random Forest (Gini importance)</h3>${bars(r.feature_importance_rf, "importance")}</div>
    <div><h3>SHAP (mean |SHAP value|)</h3>${shapBlock}</div>
  </div>`;
}

function renderEfficiency(r: Results): string {
  const rows = r.efficiency.map((e) => [
    esc(e.model), num(e.inference_time_per_sample_ms, 4), int(e.memory_footprint_kb),
    num(e.total_inference_time_s, 3), num(e.training_time_s, 1),
  ]);
  return table(["Model", "Inference / sample (ms)", "Memory (KB)", "Total inference (s)", "Training (s)"], rows) +
    `<p class="hint">Timings depend on the machine that ran the pipeline (${r.build.source === "github-actions" ? "a GitHub-hosted runner" : "a local computer"}).</p>`;
}

function renderFederated(r: Results): string {
  const f = r.federated;
  if (!f) return `<p class="muted">The federated simulation was skipped in this run.</p>`;
  const shards = f.shard_sizes.map((s, i) => [`Client ${i + 1}`, int(s)]);
  const metrics = METRICS.map(({ key, label }) => [label, num(f.metrics[key])]);
  return `<div class="two-col">
    <div><h3>Non-IID shards (Dirichlet α = ${f.non_iid_alpha})</h3>${table(["Client", "Training rows"], shards)}</div>
    <div><h3>Federated soft-voting ensemble</h3>${table(["Metric", "Test score"], metrics)}
      <p class="hint">Mean pairwise Spearman correlation of client feature importances:
      <strong>${num(f.mean_pairwise_spearman, 3)}</strong> (1 = identical rankings across clients).</p></div>
  </div>`;
}

function renderDataset(r: Results): string {
  const d = r.dataset;
  const names = Array.from(new Set([...Object.keys(d.raw_class_counts), ...Object.keys(d.class_counts_after_smote)])).sort();
  const rows = names.map((n) => [
    esc(n), int(d.raw_class_counts[n]), int(d.class_counts_after_cleaning[n]), int(d.class_counts_after_smote[n]),
  ]);
  return `<p class="intro">${int(d.raw_rows)} rows × ${d.raw_columns} columns in <code>${esc(d.file)}</code>.
      After cleaning: ${int(d.rows_after_cleaning)} rows. After SMOTE: ${int(d.train_rows + d.test_rows)} rows,
      split ${int(d.train_rows)} train / ${int(d.test_rows)} test.</p>` +
    table(["Class", "Raw", "After cleaning", "After SMOTE"], rows) +
    `<p class="hint">“—” means the class was removed or renamed during cleaning (${esc(r.settings.dropped_classes.join(", "))} dropped;
      ${esc(r.settings.zero_day_source_classes.join(", "))} → ${esc(r.settings.zero_day_label)}).</p>`;
}

function paramTable(r: Results, key: "xgboost_params" | "lightgbm_params"): string {
  const params = r.settings[key];
  const paper = PAPER_TABLE_I[key];
  const rows = Object.entries(params).map(([name, value]) => {
    const ref = paper[name];
    const changed = ref !== undefined && Number(value) !== ref;
    return [
      `<code>${esc(name)}</code>`,
      `<span class="${changed ? "changed" : ""}">${esc(value)}</span>`,
      ref === undefined ? '<span class="muted">—</span>' : changed ? `<span class="changed-note">paper: ${ref}</span>` : '<span class="muted">same as paper</span>',
    ];
  });
  return table(["Parameter", "Value used", "Paper Table I"], rows, { numericFrom: 99 });
}

function renderSettings(r: Results): string {
  const s = r.settings;
  const changedCount = (["xgboost_params", "lightgbm_params"] as const).reduce(
    (n, key) => n + Object.entries(PAPER_TABLE_I[key]).filter(([p, v]) => Number(s[key][p]) !== v).length,
    0,
  );
  const general = [
    ["Random seed", String(s.random_state)],
    ["Test split", `${s.test_size * 100}% (stratified: ${s.stratify ? "yes" : "no"})`],
    ["SMOTE k-neighbours", String(s.smote_k_neighbors)],
    ["Calibration", `${s.calibration_method}, ${s.calibration_cv}-fold ${r.stages.calibration ? "" : "(skipped)"}`],
    ["Federated clients", `${s.fl_clients} (α = ${s.fl_non_iid_alpha}) ${r.stages.federated ? "" : "(skipped)"}`],
    ["SHAP", r.stages.shap ? "on" : "skipped"],
  ].map(([a, b]) => [esc(a), esc(b)]);
  const banner = changedCount
    ? `<p class="notice">${changedCount} hyperparameter${changedCount > 1 ? "s differ" : " differs"} from the paper’s Table I (highlighted below).</p>`
    : `<p class="notice ok">All hyperparameters match the paper’s Table I.</p>`;
  return `${banner}<div class="two-col">
    <div><h3>XGBoost</h3>${paramTable(r, "xgboost_params")}</div>
    <div><h3>LightGBM</h3>${paramTable(r, "lightgbm_params")}</div>
  </div><h3>Pipeline settings</h3>${table(["Setting", "Value"], general, { numericFrom: 99 })}`;
}

function renderFooter(r: Results): void {
  const pkgs = Object.entries(r.build.packages)
    .map(([k, v]) => `${esc(k)} ${esc(v ?? "?")}`)
    .join(" · ");
  document.getElementById("footer")!.innerHTML =
    `<p>Python ${esc(r.build.python)} · ${pkgs}</p>
     <p>This page is generated from <code>results.json</code>, written by <code>scripts/run_pipeline.py</code> on every push.</p>`;
}

function wireShapToggle(r: Results): void {
  const container = document.getElementById("shap-bars");
  document.querySelectorAll<HTMLButtonElement>("button[data-shap]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const model = btn.dataset.shap!;
      document.querySelectorAll<HTMLButtonElement>("button[data-shap]").forEach((b) =>
        b.setAttribute("aria-selected", String(b === btn)),
      );
      if (container) container.innerHTML = bars(r.shap_top_features[model], "mean_abs_shap");
    });
  });
}

// ---------------------------------------------------------------- main --
function render(r: Results): void {
  renderBuild(r);
  const app = document.getElementById("app")!;
  app.innerHTML = [
    renderHeadline(r),
    section("models", "Model comparison", "Accuracy and weighted F1 on the held-out test set. The dashed line is the paper’s reported accuracy for the hybrid model.",
      renderModelChart(r) + renderMetricsTable(r)),
    section("paper", "Paper vs this run", `Final model: <strong>${esc(r.final_model)}</strong>. The paper allows small differences; see the README for why they occur.`,
      renderPaperComparison(r)),
    section("per-class", "Per-class results", `Precision, recall and F1 for each class (${esc(r.final_model)}).`, renderPerClass(r)),
    section("confusion", "Confusion matrix", `${esc(r.confusion_matrix.model)} on the test set.`, renderConfusion(r)),
    section("features", "Most important features", "", renderImportance(r)),
    section("federated", "Federated learning simulation", "", renderFederated(r)),
    section("efficiency", "Computational efficiency", "", renderEfficiency(r)),
    section("dataset", "Dataset", "", renderDataset(r)),
    section("settings", "Settings used for this run", "Read from <code>src/config.py</code> at the commit shown above. Change a value, push, and this section updates.", renderSettings(r)),
  ].join("");
  renderFooter(r);
  wireShapToggle(r);
}

async function main(): Promise<void> {
  const status = document.getElementById("status");
  try {
    const response = await fetch(`./results.json?t=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`results.json not found (HTTP ${response.status})`);
    const results = (await response.json()) as Results;
    render(results);
  } catch (error) {
    document.getElementById("paper-title")!.textContent = "";
    if (status) {
      status.innerHTML = `<strong>No results yet.</strong> ${esc((error as Error).message)}.<br />
        Run <code>python scripts/run_pipeline.py</code> to create <code>frontend/public/results.json</code>,
        or wait for the GitHub Actions pipeline to finish.`;
      status.classList.add("error");
    }
  }
}

void main();
