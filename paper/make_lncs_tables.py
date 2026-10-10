"""Derive LNCS-width table rows from the generated IEEE rows (no new numbers).
tab_main_lncs.tex = tab_main.tex with the calibration failure-cell count
(from tab_cells.tex, column 3) inserted after n_fail, and the Mondrian-by-class
columns dropped (they appear per stage as the Mond. class row of tab_platt)."""
import re, sys
g = sys.argv[1] if len(sys.argv) > 1 else "gen"
cells = {}
for line in open(f"{g}/tab_cells.tex"):
    p = [x.strip() for x in line.split("&")]
    if p[0].isdigit():
        cells[p[0]] = p[2]
out = []
for line in open(f"{g}/tab_main.tex"):
    p = line.split("&")
    st = p[0].strip()
    p.insert(3, f" {cells[st]} ")
    del p[7:9]   # Mond. class Fail/Size: same values as Rule B row of Table 6
    out.append("&".join(p))
open(f"{g}/tab_main_lncs.tex", "w").write("".join(out))
print("".join(out))

# tab_ratio_lncs.tex = tab_ratio.tex without the drive and test-window columns, so the
# table fits the LNCS text width at its natural size.
rows = []
for line in open(f"{g}/tab_ratio.tex"):
    p = line.split("&")
    del p[1:3]   # drive and test-window counts: drive counts are in the text
    rows.append("&".join(p))
open(f"{g}/tab_ratio_lncs.tex", "w").write("".join(rows))
