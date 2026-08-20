"""Generate the offline clinician annotation tool (one HTML file per annotator).

The file is fully self-contained: packets and the ICD-10 pick-list are embedded,
so it opens with file:// and needs no server, no network and no install.
Progress autosaves to localStorage, so a session can be interrupted.

It contains real patient notes and vitals. It is a LOCAL working file and must
not be published or emailed.

    python -m src.phcrx.annotate.make_tool
"""
from __future__ import annotations

import argparse
import json

from ..config import RESULTS
from ..nlp.icd_index import load_icd

OUT = RESULTS / "annotation"

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PHC clinical annotation — __ANNOTATOR__</title>
<style>
:root{--bg:#fcfcfb;--card:#fff;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a8983;
--line:#e5e4df;--accent:#2a78d6;--warn:#eb6834;--ok:#1baf7a;--radius:10px}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--card:#242423;--ink:#fff;
--ink2:#c3c2b7;--ink3:#8a8983;--line:#3a3a38}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
padding:12px 20px;z-index:10}
.bar{height:5px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:8px}
.bar>div{height:100%;background:var(--accent);width:0;transition:width .2s}
main{max-width:860px;margin:0 auto;padding:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
padding:18px 20px;margin-bottom:16px}
h1{font-size:16px;margin:0}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink2);
margin:0 0 10px}
.note{font-size:17px;padding:12px 14px;background:var(--bg);border-left:3px solid var(--accent);
border-radius:6px;white-space:pre-wrap}
.demo{color:var(--ink2);font-size:14px;margin-bottom:10px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--ink2);font-weight:600;font-size:12px;text-transform:uppercase}
.vitals{display:flex;flex-wrap:wrap;gap:8px}
.vit{background:var(--bg);border:1px solid var(--line);border-radius:6px;
padding:5px 10px;font-size:13px}
.vit b{font-weight:600}
.opt{border:2px solid var(--line);border-radius:var(--radius);padding:14px;margin-bottom:12px}
.opt.sel{border-color:var(--accent)}
.opt h3{margin:0 0 8px;font-size:14px}
label{display:block;margin:8px 0 4px;font-size:13px;color:var(--ink2)}
input,select,textarea,button{font:inherit;color:var(--ink);background:var(--card);
border:1px solid var(--line);border-radius:7px;padding:8px 10px}
input,select,textarea{width:100%}
textarea{min-height:60px;resize:vertical}
.row{display:flex;gap:10px;flex-wrap:wrap}
.row>*{flex:1;min-width:150px}
.choices{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}
.choices button{flex:1;min-width:120px;cursor:pointer;text-align:center}
.choices button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
nav{display:flex;gap:10px;align-items:center;justify-content:space-between;margin-top:18px}
nav button{cursor:pointer;padding:10px 18px}
.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.muted{color:var(--ink3);font-size:12px}
.pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;
background:var(--line);color:var(--ink2)}
datalist{max-height:200px}
.done{color:var(--ok);font-weight:600}
</style></head><body>
<header>
  <h1>PHC clinical annotation <span class="pill">__ANNOTATOR__</span>
      <span id="pos" class="muted"></span>
      <span id="saved" class="muted"></span></h1>
  <div class="bar"><div id="prog"></div></div>
</header>
<main>
  <div id="item"></div>
  <nav>
    <button id="prev">← Previous</button>
    <span id="counter" class="muted"></span>
    <div style="display:flex;gap:10px">
      <button id="export">Export answers</button>
      <button id="next" class="primary">Next →</button>
    </div>
  </nav>
  <p class="muted">Answers save automatically in this browser. Export when
  finished and return the downloaded file. Contains patient data — keep local.</p>
</main>
<script>
const PACKETS = __PACKETS__;
const ICD = __ICD__;
const ANNOTATOR = "__ANNOTATOR__";
const KEY = "phc_annot_" + ANNOTATOR;
let answers = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0;

const save = () => {
  localStorage.setItem(KEY, JSON.stringify(answers));
  const s = document.getElementById("saved");
  s.textContent = " · saved"; setTimeout(()=>s.textContent="", 1200);
};
const esc = s => String(s??"").replace(/[&<>"]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;"}[c]));

function vitalsHtml(v){
  if(!v || !v.length) return '<span class="muted">No vitals recorded</span>';
  return '<div class="vitals">' + v.map(x =>
    `<span class="vit">${esc(x.label)} <b>${esc(x.value)}</b> ${esc(x.unit)}</span>`
  ).join("") + '</div>';
}
function rxHtml(lines){
  if(!lines || !lines.length)
    return '<p><em>No medication prescribed.</em></p>';
  return '<table><tr><th>Drug</th><th>Class</th><th>Form</th><th>Dose</th>'
    + '<th>Duration</th><th>Instruction</th></tr>' + lines.map(l =>
    `<tr><td><b>${esc(l.drug)}</b></td><td class="muted">${esc(l.klass)}</td>
     <td>${esc(l.type)}</td><td>${esc(l.dose)}</td>
     <td>${esc(l.duration)}</td><td>${esc(l.instruction)}</td></tr>`).join("")
    + '</table>';
}
function choice(name, val, options, cur){
  return '<div class="choices">' + options.map(o =>
    `<button type="button" data-f="${name}" data-v="${esc(o[0])}"
      class="${cur===o[0]?'on':''}">${esc(o[1])}</button>`).join("") + '</div>';
}

function render(){
  const p = PACKETS[i];
  const a = answers[p.ann_id] || {};
  let h = `<div class="card">
    <h2>Encounter ${esc(p.ann_id)} · ${p.task === "icd" ? "Diagnosis coding" : "Prescription review"}</h2>
    <div class="demo">Age <b>${esc(p.age)}</b> · Sex <b>${esc(p.sex)}</b></div>
    <div class="note">${esc(p.note) || "<em>(no complaint recorded)</em>"}</div>
    <div style="margin-top:12px">${vitalsHtml(p.vitals)}</div>
  </div>`;

  if(p.task === "icd"){
    h += `<div class="card">
      <h2>Assign ICD-10 (up to 3, most important first)</h2>
      <p class="muted">Type to search by code or description. Leave blank if no
      code is appropriate.</p>
      <div class="row">
        <div><label>Primary</label>
          <input list="icdlist" data-f="icd1" value="${esc(a.icd1||"")}" placeholder="e.g. I10"></div>
        <div><label>Secondary</label>
          <input list="icdlist" data-f="icd2" value="${esc(a.icd2||"")}"></div>
        <div><label>Tertiary</label>
          <input list="icdlist" data-f="icd3" value="${esc(a.icd3||"")}"></div>
      </div>
      <label>No code is appropriate for this note</label>
      ${choice("no_code", a.no_code, [["yes","No code appropriate"],["no","A code applies"]], a.no_code)}
      <label>How confident are you in this coding?</label>
      ${choice("confidence", a.confidence, [["high","High"],["medium","Medium"],["low","Low"]], a.confidence)}
      <label>Is the note codable at all, or too vague / multi-problem?</label>
      ${choice("codability", a.codability, [["clear","Clear single problem"],
        ["multi","Multiple problems"],["vague","Too vague to code"]], a.codability)}
      <label>Comments (optional)</label>
      <textarea data-f="comment">${esc(a.comment||"")}</textarea>
    </div>`;
  } else {
    h += `<div class="card"><h2>Two prescriptions were written for this encounter</h2>
      <p class="muted">They are shown in random order. Judge them on clinical
      appropriateness for this patient.</p>
      <div class="opt"><h3>Option 1</h3>${rxHtml(p.option1)}</div>
      <div class="opt"><h3>Option 2</h3>${rxHtml(p.option2)}</div>
    </div>
    <div class="card">
      <h2>Which is more clinically appropriate?</h2>
      ${choice("preference", a.preference, [["1","Option 1"],["2","Option 2"],
        ["equal","Equivalent"],["both_bad","Both inappropriate"]], a.preference)}
      <label>Option 1 — safety</label>
      ${choice("safety1", a.safety1, [["appropriate","Appropriate"],
        ["suboptimal","Suboptimal but safe"],["unsafe","Unsafe / harmful"]], a.safety1)}
      <label>Option 2 — safety</label>
      ${choice("safety2", a.safety2, [["appropriate","Appropriate"],
        ["suboptimal","Suboptimal but safe"],["unsafe","Unsafe / harmful"]], a.safety2)}
      <label>Would you have prescribed something materially different?</label>
      ${choice("would_differ", a.would_differ, [["no","No"],["yes","Yes"]], a.would_differ)}
      <label>Comments (optional)</label>
      <textarea data-f="comment">${esc(a.comment||"")}</textarea>
    </div>`;
  }
  document.getElementById("item").innerHTML = h;

  document.querySelectorAll("[data-f]").forEach(el => {
    if(el.tagName === "BUTTON"){
      el.onclick = () => {
        const f = el.dataset.f;
        answers[p.ann_id] = answers[p.ann_id] || {};
        answers[p.ann_id][f] = el.dataset.v;
        save(); render();
      };
    } else {
      el.oninput = () => {
        answers[p.ann_id] = answers[p.ann_id] || {};
        answers[p.ann_id][el.dataset.f] = el.value;
        save();
      };
    }
  });

  const done = Object.keys(answers).length;
  document.getElementById("prog").style.width = (100*done/PACKETS.length)+"%";
  document.getElementById("counter").textContent =
    `Item ${i+1} of ${PACKETS.length} · ${done} answered`;
  document.getElementById("pos").textContent =
    ` · ${done === PACKETS.length ? "complete" : done + "/" + PACKETS.length}`;
}

document.getElementById("next").onclick = () => { if(i<PACKETS.length-1){i++;render();} };
document.getElementById("prev").onclick = () => { if(i>0){i--;render();} };
document.onkeydown = e => {
  if(e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
  if(e.key === "ArrowRight") document.getElementById("next").click();
  if(e.key === "ArrowLeft") document.getElementById("prev").click();
};
document.getElementById("export").onclick = () => {
  const blob = new Blob([JSON.stringify(
    {annotator: ANNOTATOR, completed: Object.keys(answers).length,
     total: PACKETS.length, answers}, null, 1)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "annotations_" + ANNOTATOR + ".json";
  a.click();
};
render();
</script>
<datalist id="icdlist">__ICDOPTIONS__</datalist>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotators", nargs="+", default=None,
                    help="defaults to the assignment in the key file")
    args = ap.parse_args()

    packets = json.loads((OUT / "packets.json").read_text(encoding="utf-8"))
    key = json.loads((OUT / "KEY_do_not_share_with_annotators.json").read_text())
    assignment = key["assignment"]
    by_id = {p["ann_id"]: p for p in packets}

    icd = load_icd()
    icd_pairs = [(r["id"], r["descr"]) for _, r in icd.iterrows()]
    options = "".join(
        f'<option value="{c}">{c} — {d}</option>' for c, d in icd_pairs)
    icd_json = json.dumps({c: d for c, d in icd_pairs})

    names = args.annotators or sorted(assignment)
    for name in names:
        ids = assignment.get(name, [])
        subset = [by_id[i] for i in ids if i in by_id]
        html = (TEMPLATE
                .replace("__PACKETS__", json.dumps(subset, ensure_ascii=False))
                .replace("__ICD__", icd_json)
                .replace("__ICDOPTIONS__", options)
                .replace("__ANNOTATOR__", name))
        path = OUT / f"annotate_{name}.html"
        path.write_text(html, encoding="utf-8")
        print(f"{path.name:28s} {len(subset):4d} items  {path.stat().st_size/1024:.0f} KB")

    print(f"\nOpen each file in a browser (double-click). Offline, no install.")
    print("Contains patient data — keep local, do not email or publish.")


if __name__ == "__main__":
    main()
