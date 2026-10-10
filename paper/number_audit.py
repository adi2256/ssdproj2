"""
Every number printed in the paper, traced to a source.

    python paper/number_audit.py PAPER.pdf RESULTS_DIR GEN_DIR

Collects every decimal and every number >= 100 from the PDF text (body only,
references excluded) and looks for it in: the generated macros and table rows
(GEN_DIR), every value in the released result files and prediction-derived
summaries (RESULTS_DIR, rounded to 1-3 decimals and as percentages), and the
manifest summaries. Numbers with no source are printed for manual review;
known constants (years, settings stated in the text) are listed separately.
"""
import glob, json, re, subprocess, sys
import polars as pl

pdf, res, gen = sys.argv[1:4]
extra = sys.argv[4:]          # more folders with source files, e.g. reports/
text = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout
text = text[:text.rfind("References")]
toks = re.findall(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d+|\d{3,})(?![\w])", text)

pool = set()
def add(v):
    try:
        x = float(str(v).replace(",", "").replace("{", "").replace("}", ""))
    except ValueError:
        return
    for d in (0, 1, 2, 3, 4):
        pool.add(f"{x:.{d}f}"); pool.add(f"{100*x:.{d}f}")
for v in json.load(open(f"{gen}/numbers.json")).values():
    add(v)
for f in glob.glob(f"{gen}/tab_*.tex"):
    for m in re.findall(r"\d[\d{},]*\.?\d*", open(f).read()):
        add(m.replace("{,}", ""))
src_dirs = [res] + extra
for f in [g for d in src_dirs for g in glob.glob(f"{d}/**/*.csv", recursive=True) + glob.glob(f"{d}/**/*.csv.gz", recursive=True)]:
    if "drive" in f:
        continue
    try:
        df = pl.read_csv(f, infer_schema_length=10000)
    except Exception:
        continue
    for c in df.columns:
        if df[c].dtype.is_numeric():
            for v in df[c].drop_nulls().unique().to_list():
                add(v)
for f in glob.glob(f"{res}/**/*.json", recursive=True):
    def walk(o):
        if isinstance(o, dict): [walk(v) for v in o.values()]
        elif isinstance(o, list): [walk(v) for v in o]
        else: add(o)
    try:
        walk(json.load(open(f)))
    except Exception:
        pass

for d in extra:
    for f in glob.glob(f"{d}/**/*.txt", recursive=True):
        for m in re.findall(r"\d[\d,]*\.?\d*", open(f).read()):
            add(m.replace(",", ""))
# constants stated in the text itself: SMART IDs, years, figure axis ticks
CONST = {"197", "199", "183", "184", "187", "2018", "2019", "10000", "20000",
         "30000", "40000"}
missing = []
for t in toks:
    k = t.replace(",", "")
    if k in CONST:
        continue
    if k not in pool and f"{float(k):.3f}" not in pool:
        missing.append(t)
print(f"{len(toks)} numbers in the body; {len(missing)} without an automatic source:")
for m in sorted(set(missing), key=lambda s: float(s.replace(',', ''))):
    print("  ", m)
