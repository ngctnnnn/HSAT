#!/usr/bin/env python3
"""Gom F1/AUC từ logs/table2/*.log -> table2_results.csv theo đúng layout sheet 'Table 2'.
Sửa 2 regex dưới cho khớp format print của run_gdn_defense.py / run_usad_defense.py."""
import re, glob, csv, os, statistics as st

LOGDIR = os.path.expanduser("~/maikhanh/HSATproject-main/logs/table2")
# khớp "f1= 0.0", "auc_score= 0.5", "F1: 0.03", "AUC 0.52"...
F1_RE  = re.compile(r"\bf1[\w ]*?[:=]?\s*([0-9]*\.?[0-9]+)", re.I)
AUC_RE = re.compile(r"\bauc[\w ]*?[:=]?\s*([0-9]*\.?[0-9]+)", re.I)

DATASETS = ["PEMS03", "Weather", "ETTm1"]; TRIGGERS = ["backtime", "msat", "hsat"]
DETS = ["gdn", "usad"]; SEEDS = [1, 2, 3]
LABEL = {"backtime": "Backtime", "msat": "MSAT", "hsat": "HSAT"}

def last(rx, txt):
    m = rx.findall(txt); return float(m[-1]) if m else None

res = {}
for f in glob.glob(f"{LOGDIR}/*.log"):
    det, tr, ds, seed = re.match(r"(gdn|usad)_(\w+?)_(\w+?)_seed(\d+)\.log", os.path.basename(f)).groups()
    txt = open(f).read()
    res[(det, tr, ds, int(seed))] = (last(F1_RE, txt), last(AUC_RE, txt), "END" in txt)

rows = [["Dataset", "Trigger"] + [f"{d.upper()} {m} Run{s}" for d in DETS for m in ("F1", "AUC") for s in SEEDS + ["Avg"]]]
for ds in DATASETS:
    for tr in TRIGGERS:
        row = [ds, LABEL[tr]]
        for det in DETS:
            for mi in (0, 1):
                vals = [res.get((det, tr, ds, s), (None, None, False))[mi] for s in SEEDS]
                row += ["" if v is None else f"{v:.4f}" for v in vals]
                ok = [v for v in vals if v is not None]
                row.append(f"{st.mean(ok):.4f}" if ok else "")
        rows.append(row)
missing = [k for k in [(d, t, ds, s) for d in DETS for t in TRIGGERS for ds in DATASETS for s in SEEDS] if k not in res or not res[k][2]]
with open("table2_results.csv", "w", newline="") as fo: csv.writer(fo).writerows(rows)
for r in rows: print("\t".join(map(str, r)))
print(f"\n{len(res)}/54 log; chưa xong/thiếu: {missing if missing else 'không'}")
