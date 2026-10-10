"""Build the point-by-point response letter (ReportLab Platypus)."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

F = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", F + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVB", F + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVI", F + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DVM", F + "DejaVuSansMono.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI",
                   boldItalic="DVB")

REPO = "https://github.com/adi2256/ssdproj2"
ACCENT = colors.HexColor("#1f3b63")
GREY = colors.HexColor("#555555")
RULE = colors.HexColor("#c8ced8")
SHADE = colors.HexColor("#f2f4f7")

base = ParagraphStyle("base", fontName="DV", fontSize=9.2, leading=12.6,
                      alignment=TA_LEFT)
title = ParagraphStyle("title", parent=base, fontName="DVB", fontSize=15,
                       leading=19, textColor=ACCENT, spaceAfter=4)
sub = ParagraphStyle("sub", parent=base, fontSize=9.5, textColor=GREY)
h1 = ParagraphStyle("h1", parent=base, fontName="DVB", fontSize=12,
                    leading=16, textColor=ACCENT, spaceBefore=10,
                    spaceAfter=5, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=base, fontName="DVB", fontSize=9.8,
                    leading=13, spaceBefore=8, spaceAfter=3)
quote = ParagraphStyle("quote", parent=base, fontName="DVI", textColor=GREY,
                       leftIndent=8)
body = ParagraphStyle("body", parent=base, spaceAfter=3)
small = ParagraphStyle("small", parent=base, fontSize=8, leading=10.4)
smallb = ParagraphStyle("smallb", parent=small, fontName="DVB")
bullet = ParagraphStyle("bullet", parent=base, leftIndent=12,
                        bulletIndent=2, spaceAfter=1.5)


def code(s):
    return f'<font name="DVM" size="8.2">{s}</font>'


def P(t, s=body):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(i, bullet, bulletText="•") for i in items]


def item(tag, comment, response, where=None, evidence=None):
    """One reviewer point: quoted comment, response, location, evidence."""
    rows = [[P(f"<b>{tag}</b>", small), P(comment, quote)],
            [P("<b>Response</b>", small), P(response, body)]]
    if where:
        rows.append([P("<b>Manuscript</b>", small), P(where, body)])
    if evidence:
        rows.append([P("<b>Repository</b>", small), P(evidence, body)])
    t = Table(rows, colWidths=[22 * mm, 150 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), SHADE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]))
    t.splitByRow = 1
    return [t, Spacer(1, 5)]


def grid(data, widths, header=True):
    rows = [[P(c, smallb if (header and i == 0) else small) for c in r]
            for i, r in enumerate(data)]
    t = Table(rows, colWidths=[w * mm for w in widths], repeatRows=1)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("GRID", (0, 0), (-1, -1), 0.4, RULE),
          ("TOPPADDING", (0, 0), (-1, -1), 2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), SHADE))
    t.setStyle(TableStyle(st))
    return t


story = []
story += [
    P("Response to Reviewers", title),
    P("<b>Marginal Coverage Is Not Enough: Auditing Conformal Failure "
      "Prediction on a Heterogeneous SSD Fleet</b>", sub),
    P("Aditya Aryan Singh, Ananshya, Ruby D · School of Computer Science and "
      "Engineering, VIT Vellore · 11 October 2026", sub),
    Spacer(1, 8),
    P("We thank the reviewer for a careful reading. Every comment is "
      "answered below in the order it appears in the annotated manuscript. "
      "For each we give our response, where the change is in the revised "
      "manuscript (Springer LNCS format, 15 pages), and the repository file "
      "that supports it. Section, table and figure numbers refer to the "
      "revised manuscript; the reviewer's annotated copy used an earlier "
      "numbering (its Table 1 is now Table 2, its Table 3 is now Table 4, "
      f"its Table 5 is now Table 6). Repository: {REPO} (branch "
      f"{code('main')}). Paths below are relative to "
      f"{code('paper_results/final/')} unless they start with a top-level "
      "folder."),
    P("Part A answers the reviewer's comments. Part B answers the points "
      "from a follow-up check of the revised PDF. Part C states plainly what "
      "was not done."),
]

story.append(P("Summary of changes", h1))
story += bullets([
    "<b>Full rerun on a deterministic sampler.</b> The sampler now sorts "
    "drives by (model, serial) before drawing and prints a fingerprint; every "
    "downstream ordering is seeded. All reported results come from one batch "
    f"run of {code('notebooks/revision_batch.py')} on this sampler, "
    "including the three sampling ratios, ten seeds, all robustness checks "
    "and the three held-out-vendor folds.",
    "<b>Released manifests.</b> Sampled drive identifiers, fingerprints, "
    "split assignments for every split used, seeds and the configuration of "
    "every experiment. Two independent Kaggle sessions produced identical "
    "manifests.",
    "<b>Released predictions and a provenance table.</b> Per-window "
    "predictions are released for the main, ratio, within-vendor and "
    "within-model runs. New Table 1 marks each result P (recomputable from "
    "released predictions) or R (released result file, full rerun needed).",
    "<b>Guaranteed versus measured.</b> The formal guarantee is stated for "
    "the drive (the exchangeable unit). Window-level, per-stratum and "
    "held-out-vendor values are labelled as empirical. A new drive-level "
    "check (one window per drive) reports the guaranteed quantity beside "
    "the window-level one.",
    "<b>Causal restraint.</b> The vendor–age result is described as "
    "evidence consistent with confounding in this dataset, not proof of a "
    "cause.",
    "<b>Number tracing.</b> Every in-text number is a macro generated from "
    "the released files, and a script traces all 540 numbers in the PDF "
    "back to a released file.",
])

# ---------------------------------------------------------------- Part A
story.append(P("Part A. Reviewer comments", h1))

story += item(
    "A1 · p. 1",
    "“The Abstract gives several headline results that must match the final "
    "regenerated tables … Rerun the corrected pipeline first. Then "
    "synchronize the Abstract, Introduction, Results and Conclusion. In "
    "particular check 0.330, 0.828, 0.89 and the held-out vendor C value "
    "0.56/0.556. Do not force values to match without verifying outputs.”",
    "Done. The corrected pipeline was rerun first, as one batch, and the "
    "manuscript was then regenerated from its outputs. No result number is "
    f"typed by hand: {code('notebooks/paper_tables.py')} writes every table "
    "and a macro file of in-text numbers from the released outputs, and the "
    f"manuscript uses only those macros. {code('paper/number_audit.py')} "
    "then extracts all 540 numbers from the built PDF and looks each up in "
    "the released files. Every number matched except 475,056, the dataset's "
    f"drive count, which equals the sum of {code('n_drives')} in "
    f"{code('reports/vendor_profile.csv')} (checked by hand). The four "
    "values named by the reviewer now read: oldest-stage failure coverage "
    "<b>0.330</b> (CI 0.276–0.388); joint Mondrian <b>0.828</b> "
    "(CI 0.780–0.902); ten-seed mean <b>0.891</b> (was “0.89”); held-out "
    "vendor C marginal coverage <b>0.556</b> in every occurrence (the "
    "abstract previously said 0.56). Coverage values use three decimals "
    "throughout.",
    "Abstract; Sect. 1 contributions; Sect. 5.1–5.5; Sect. 6.",
    f"{code('ratio_summary.csv')}, {code('seeds.csv')}, "
    f"{code('shift_rank.csv')}, {code('preds_r2p7_*.parquet')}; "
    f"{code('paper/number_audit.py')}.")

story += item(
    "A2 · p. 1",
    "“Holding out a vendor breaks even marginal coverage” could be read as "
    "a theoretical guarantee being tested under ordinary assumptions … Label "
    "the held-out-vendor numbers as empirical fold results. State that the "
    "ordinary split-conformal guarantee does not automatically apply when "
    "calibration and test populations are not exchangeable. Reconcile 0.56 "
    "with 0.556.”",
    "Agreed and changed. The abstract now reads: “On one held-out-vendor "
    "fold, where calibration and test drives are not exchangeable, observed "
    "marginal coverage was 0.556.” Contribution 4 and Sect. 5.5 say the "
    "same. Sect. 5.5 now opens the held-out-vendor results with: “These "
    "are observed results on single folds; the split-conformal guarantee "
    "does not apply, because calibration and test drives come from "
    "different vendors and are not exchangeable.” Sect. 4 lists every "
    "held-out-vendor result under “Measured, not guaranteed”. The number "
    "is 0.556 everywhere.",
    "Abstract; Sect. 1 (contribution 4); Sect. 4 “What is guaranteed and "
    "what is measured”; Sect. 5.5.",
    f"{code('shift_rank.csv')}, {code('shift_cells/')}.")

story += item(
    "A3 · p. 2",
    "“The age/vendor relationship is observational; the paper should not "
    "imply that the analysis proves one exclusive causal explanation … "
    "Describe the result as evidence consistent with vendor–age confounding "
    "in this dataset. Verify Cramér's V (reported as 0.65/0.654) after "
    "rerunning and use consistent rounding.”",
    "Agreed. Contribution 2 is now titled “Evidence of a covariate–vendor "
    "confound” and is limited to “here the vendors occupy largely disjoint "
    "age ranges.” The abstract says the pattern “is consistent with "
    "vendor–age confounding and does not support a uniform wear effect”; "
    "Sect. 5.3 says the two within-vendor checks “are consistent with, but "
    "do not establish, the confounding explanation”; the conclusion says "
    "“evidence consistent with vendor–age confounding in this dataset, not "
    "proof of a single cause.” Algorithm 2's final step now names "
    "population composition as “a candidate explanation”. Cramér's V was "
    "recomputed from the rerun's test predictions and is 0.654 in every "
    "occurrence.",
    "Abstract; Sect. 1; Alg. 2; Sect. 5.3; Sect. 6.",
    f"{code('preds_r2p7_test.parquet')} (recomputed by "
    f"{code('notebooks/paper_tables.py')}, {code('confound_stats')}).")

story += item(
    "A4 · p. 4",
    "“The drive-level exchangeability guarantee and the window-level "
    "empirical estimates are different evaluation units … Keep the caveat "
    "explicit wherever results are summarized.”",
    "Agreed. A paragraph “Guaranteed unit versus reported unit” in Sect. 3 "
    "explains why dependent windows from drives of unequal history do not "
    "inherit the guarantee. The abstract now says: “The formal guarantee is "
    "stated for drives; the coverage values we report are empirical "
    "window-level estimates.” To show the guaranteed quantity directly we "
    "added a drive-level check (one randomly drawn window per calibration "
    "and per test drive, 200 draws): marginal coverage 0.916 "
    "(0.912–0.920 across draws), failure-class coverage 0.802 (Sect. 5.1).",
    "Abstract; Sect. 3; Sect. 4; Sect. 5.1; Sect. 6.",
    f"{code('notebooks/drive_level_check.py')} (runs on the released "
    "predictions alone).")

story += item(
    "A5 · p. 4",
    "“The Abstract, Results and Conclusion must not accidentally overstate "
    "the guarantee … State the assumptions for split conformal and for each "
    "Mondrian cell; label subgroup/window values as measured.”",
    "We checked every coverage claim. Sect. 4 “What is guaranteed and what "
    "is measured” states the assumption for split conformal (exchangeable "
    "calibration and test drives, one window per drive) and for a Mondrian "
    "cell (calibration and test drives exchangeable within the cell, at "
    "least 9 calibration points; fallback cells carry no cell guarantee; "
    "nothing transfers to a group absent from calibration). It then lists "
    "what is measured, not guaranteed: every window-level value, every "
    "per-class, per-vendor and per-stage figure for split conformal, and "
    "every held-out-vendor result. Sect. 5.4 repeats the cell-level "
    "condition at the point where the repair result is given. Subgroup, "
    "window-level and held-out-vendor values are called “observed” or "
    "“empirical” in the abstract, Results, figure captions and conclusion.",
    "Sect. 4; Sect. 5.4; Fig. 4 caption; Sect. 6.")

story += item(
    "A6 · p. 6",
    "“A seed alone did not make the earlier sampling reproducible because "
    "the input list order could vary … Release the exact sampled "
    "healthy-drive IDs, deterministic ordering/fingerprint, and "
    "train/calibration/test drive IDs. Record sampling, split and model "
    "seeds plus each experiment's configuration.”",
    "Done. The Reproducibility paragraph now states the earlier defect "
    "openly. The sampler sorts the drive list by (model, serial) before "
    f"{code('DataFrame.sample(seed)')} and prints a SHA1 fingerprint of the "
    "drawn keys. The drive table, split permutation, window construction, "
    "training-window cap and random forest are seeded and order-fixed. For "
    "every drive sample the repository releases: all sampled drive "
    "identifiers with failure label and vendor; the split part of every "
    "drive for every split used (standard, within-vendor, within-model, "
    "each held-out vendor, each held-out wear stage); the fingerprint and "
    "labelling counts; and the configuration of every experiment.",
    "Sect. 4, Reproducibility paragraph; Table 1.",
    f"{code('manifests/drives_&lt;tag&gt;.csv.gz')}, "
    f"{code('manifests/samples.json')}, "
    f"{code('manifests/experiments.json')}, "
    f"{code('manifests/README.md')}; sampler in "
    f"{code('notebooks/kaggle_setup.py')}.")

story += item(
    "A7 · p. 6",
    "“The seed list does not by itself identify the exact sample and split "
    "used for every experiment … Add per-run manifests … Confirm each "
    "reported run used the corrected sampler.”",
    "Done. Manifests exist for the three ratio samples (seed 42) and for "
    "seeds 0–8. Confirmation that the reported runs used the corrected "
    "sampler: (i) the main sample's fingerprint in the manifest "
    f"({code('ee3ce82fe326')}) is the value printed by the batch run that "
    "produced the released predictions; (ii) for all 12 calibration and "
    "test prediction files of the ratio and within-vendor runs, the "
    "manifest has exactly the drives in the file, each in its recorded "
    "split part, every window maps to a drive of the recorded vendor, and "
    "every positive window maps to a failed drive (12/12 pass); (iii) the "
    "wear-stage hold-out manifest reproduces the per-stage drive counts of "
    "the released robustness run (19,347 / 5,181 / 11,980 / 9,541 / "
    "11,726); (iv) two independent sessions produced identical manifests. "
    "The repository README has an artifact index giving, for every table, "
    "figure and in-text result, its file, manifest column and seed.",
    "Sect. 4 Reproducibility; Table 1.",
    f"{code('manifests/consistency_check.csv')}, "
    f"{code('manifests/cross_session_check.csv')}, "
    f"{code('manifests/samples.json')}; artifact index in "
    f"{code('README.md')}.")

story += item(
    "A8 · p. 6",
    "“This explicitly says the transfer diagnostics are from an earlier run "
    "… Rerun all three held-out-vendor folds with the corrected "
    "deterministic procedure. If impossible, mark those numbers as "
    "earlier-run results not independently reproduced.”",
    "Done. All three held-out-vendor folds were rerun in the final batch on "
    "the corrected sampler, and the transfer diagnostics (range, scale and "
    "correlation-sign) were recomputed on the same 60,000-drive seed-42 "
    "sample. The exception clause was removed. The Sect. 5.5 text uses the "
    "recomputed values. Two earlier statements did not hold on the "
    "recomputation and were corrected: that SMART 12 correlates with "
    "failure in the opposite direction for vendor B (its sign is not "
    "stable, positive in 22% of resamples), and that vendor C's counters "
    "spread wider than the training data (vendor A's do, 33× the training "
    "standard deviation for SMART 199, while C's are narrower, 0.02×). The "
    "SMART 9 sign result held (B negative, A and C positive, in all 500 "
    "drive-clustered resamples). These results are marked R "
    "in Table 1, with the verification caveat in its caption (see C1).",
    "Sect. 5.5; Table 1; Sect. 6.",
    f"{code('shift_rank.csv')}, {code('shift_cells/')}, "
    f"{code('manifests/transfer_diagnostics.csv')}, "
    f"{code('manifests/transfer_sign_bootstrap.csv')}; folds in "
    f"{code('manifests/drives_r2p7.csv.gz')}, columns "
    f"{code('part_holdout_vendor_{A,B,C}')}.")

story += item(
    "A9 · p. 7",
    "“Table 1 [now Table 2] … Regenerate the table from saved outputs; verify "
    "coverage, failure coverage, calibration-cell counts, intervals and set "
    "sizes.”",
    "Done. Table 2 is generated from the released main-run predictions by "
    f"{code('notebooks/paper_tables.py')}; intervals come from the "
    "drive-clustered bootstrap, which recomputes every conformal threshold "
    "from the resampled calibration set and was checked to reproduce the "
    "released prediction sets exactly. Calibration cell counts are now "
    "given exactly: 221–711 failure windows per stage, and 29,750–31,636 "
    "calibration and 37,166–39,212 test healthy windows per stage. Status "
    "P in Table 1.",
    "Table 2; Sect. 5.4.",
    f"{code('preds_r2p7_{cal,test}.parquet')}, "
    f"{code('boot_pooled.csv')}, {code('meta_r2p7.json')}.")

story += item(
    "A10 · p. 8",
    "“Table 3 [now Table 4] … Reproduce 1:1, 2.68:1 and 5:1 from scratch. "
    "Confirm failed-drive counts and reported values from logs/artifacts, "
    "not from the current PDF.”",
    "Done. Each ratio is a full rerun on the corrected sampler (sampling, "
    "split, training, calibration) at seed 42, with manifests and "
    f"fingerprints ({code('63f59e1ae572')}, {code('ee3ce82fe326')}, "
    f"{code('443636a9f21a')} for 32,610, 60,000 and 97,830 drives). Test "
    "failed-drive counts (2,872 / 2,858 / 2,884) and every value in the "
    "table are computed from the released prediction files, not copied "
    "from the PDF. The set-size column is now headed “Size (all)” and "
    "the caption defines it as mean set size over all stages.",
    "Table 4; Sect. 5.2.",
    f"{code('preds_{r1,r2p7,r5}_{cal,test}.parquet')}, "
    f"{code('ratio_summary.csv')}, "
    f"{code('manifests/drives_{r1,r2p7,r5}.csv.gz')}.")

story += item(
    "A11 · p. 9",
    "“Table 5 [now Table 6] supports a dataset-specific within-vendor "
    "observation, not a universal age or vendor law … Recompute estimates "
    "and drive-clustered intervals from corrected runs. Describe the "
    "directions as observed for these vendors and these data.”",
    "Done. Table 6 and Fig. 3 are recomputed from the rerun's within-vendor "
    "predictions with drive-clustered intervals. The Fig. 3 caption now "
    "says “In these data the direction differs between vendors, so the fleet-wide "
    "age pattern is not reproduced within vendors,” and the Scope paragraph "
    "says “vendor A” means that vendor's drives in this release.",
    "Table 6; Fig. 3; Sect. 5.3; Sect. 1 Scope.",
    f"{code('preds_wv{A,B,C}_{cal,test}.parquet')}, "
    f"{code('boot_wv.csv')}; split in "
    f"{code('manifests/drives_r2p7.csv.gz')}, column "
    f"{code('part_within_vendor')}.")

story += item(
    "A12 · p. 10",
    "“Different trends weaken a simple fleet-wide causal age "
    "interpretation, but they do not prove a single alternative cause … Say "
    "the pattern is consistent with vendor–age confounding and avoid "
    "claiming vendor alone is proven to cause the shortfall.”",
    "Agreed; see A3. The within-vendor and within-model checks are "
    "described as “consistent with, but do not establish, the confounding "
    "explanation.” No sentence attributes the shortfall to vendor alone.",
    "Sect. 5.3; Fig. 3 caption; Sect. 6.")

story += item(
    "A13 · p. 11",
    "“An interval reaching 0.90 does not itself prove a formal guarantee … "
    "Call 0.828 the observed oldest-stage estimate, and explain the "
    "assumptions needed for any cell-level guarantee. Do not imply "
    "unseen-vendor equivalence.”",
    "Agreed. Sect. 5.4 now reads: “With calibration drives from the same "
    "three-vendor fleet … the observed window-level failure coverage rises "
    "from 0.330 to 0.828 (0.780–0.902) … This is an empirical result, not "
    "a guarantee: a cell-level guarantee would need exchangeable "
    "calibration and test drives within the (stage, failure) cell, one "
    "window per drive, and implies nothing for a vendor absent from "
    "calibration.” The Fig. 4 caption adds “an empirical estimate, "
    "same-fleet calibration”. The text also states that the gain comes "
    "from larger, more ambiguous sets, not from better discrimination.",
    "Sect. 5.4; Fig. 4 caption.",
    f"{code('preds_r2p7_{cal,test}.parquet')}, {code('boot_pooled.csv')}.")

story += item(
    "A14 · p. 12",
    "“Holding out vendor C breaks the marginal guarantee itself” can "
    "confuse an observed failure of coverage with a violation of the "
    "guarantee … Say observed marginal coverage was 0.556 on the held-out-C "
    "fold under distribution shift … Rerun and verify all three folds.”",
    "Agreed. The sentence now reads “Holding out C, observed marginal "
    "coverage is 0.556, with 43.4% empty sets,” after the statement that "
    "the guarantee does not apply when calibration and test drives are not "
    "exchangeable. All three folds were rerun (A8).",
    "Sect. 5.5.",
    f"{code('shift_rank.csv')}.")

story += item(
    "A15 · p. 13",
    "“The conclusion must not generalize the three observed vendors to all "
    "vendors or imply a proven single causal mechanism … state that the "
    "results do not guarantee performance for a previously unseen vendor.”",
    "Agreed. The Threats paragraph opens: “The vendor results are three "
    "case studies on one fleet.” The conclusion opens “On one SSD fleet”, "
    "describes the confound as “evidence consistent with vendor–age "
    "confounding in this dataset, not proof of a single cause”, qualifies "
    "the repair as “with same-fleet calibration”, and states “None of this "
    "guarantees coverage for an unseen vendor.”",
    "Sect. 6.")

story += item(
    "A16 · p. 13",
    "“0.330 to 0.828; 0.89 over ten seeds” … Recalculate from corrected "
    "artifacts, then update every occurrence.”",
    "Done; see A1. All occurrences are generated from one macro each, so "
    "they cannot disagree. Ten-seed values: split-conformal oldest-stage "
    "failure coverage 0.271–0.349 (mean 0.309); joint Mondrian 0.828–0.933 "
    "(mean 0.891).",
    "Abstract; Sect. 5.2; Sect. 5.4; Sect. 6.",
    f"{code('seeds.csv')}; seed manifests "
    f"{code('manifests/drives_seed{0..8}.csv.gz')}.")

story += item(
    "A17 · p. 14",
    "Reference [4] (Vovk, Gammerman, Shafer): verify edition, year, "
    "publisher and location.",
    "Verified against the publisher's record (link.springer.com, "
    "DOI 10.1007/978-3-031-06649-8): <i>Algorithmic Learning in a Random "
    "World</i>, 2nd edn., Springer, Cham, 2022, authors Vladimir Vovk, "
    "Alexander Gammerman, Glenn Shafer. The entry matches and now carries "
    "the DOI.",
    "Reference [4].")

story += item(
    "A18 · p. 14",
    "Tursunbadalov preprint: verify authors, title, year and arXiv ID; "
    "compare versions before changing reference numbers.",
    "Verified against arXiv:2607.06605 (v1, submitted 7 July 2026; authors "
    "Muhammadjon Tursunbadalov and Mustafojon Tursunbadalov; title as "
    "printed). Both authors share the initial M., so the entry keeps full "
    "given names to keep them distinguishable. On numbering: references are "
    "numbered by first citation. In the revised manuscript this preprint is "
    "[16], cited once in Sect. 2, and the dataset is [17]. The comment that "
    "[17] is the preprint appears to refer to a different version of the "
    "manuscript; in the version reviewed and in this revision the numbering "
    "is as stated.",
    "Reference [16]; Sect. 2.")

story += item(
    "A19 · p. 14",
    "Alibaba dataset [17]: verify the URL/access date and all in-text "
    "citations; keep dataset and preprint at the correct numbers.",
    "Checked. [17] is the dataset (dcbrain repository, "
    f"{code('ssd_smart_logs')} directory, last accessed 10 October 2026). "
    "It is cited in Sect. 3 together with the paper that released it, as "
    "[1,17]. All 17 references are cited in the text, in first-citation "
    "order, and every citation resolves to the intended entry.",
    "Sect. 3; Reference [17].")

# ---------------------------------------------------------------- Part B
story.append(P("Part B. Follow-up check of the revised PDF", h1))
story.append(P("A follow-up check of the revised PDF raised the points below. "
               "Points it marked as already satisfied (theory versus "
               "empirical unit; causal restraint in the conclusion; "
               "reference [4]) were kept as they are and are not repeated."))

story += item(
    "B1",
    "Reconcile set size 1.619 (main run) versus 1.633 (sampling-ratio "
    "table); clarify which run each headline value refers to.",
    "Both values come from the same main run (60,000 drives, seed 42; the "
    "2.68:1 row of Table 4). They measure different populations: 1.619 is "
    "the mean set size on the oldest wear stage; 1.633 is the mean over "
    "all stages. The abstract and conclusion now say “1.619 of 2 on that "
    "stratum”; Sect. 5.4 gives both (“1.619 on that stage … 1.633 over "
    "all stages, Table 4”); Table 4's column is headed “Size (all)”.",
    "Abstract; Sect. 5.4; Table 4; Sect. 6.")

story += item(
    "B2",
    "Introduction reports 1.1% and 0.6% failure coverage for held-out "
    "vendors A/B and 0.556 marginal coverage for C. Keep the same precision "
    "and distinguish failure coverage from marginal coverage.",
    "Changed to three-decimal values, 0.011 and 0.006, labelled "
    "“failure coverage”, beside the marginal coverage 0.556, in both the "
    "Introduction and Sect. 5.5.",
    "Sect. 1 (contribution 4); Sect. 5.5.")

story += item(
    "B3",
    "Confirm the repository contains sampled IDs and per-experiment seeds "
    "for every experiment, linked unambiguously.",
    "The README's artifact index lists every table, figure and in-text "
    "result with its released file, manifest file and column, seed, and "
    "P/R status. Gaps are listed there rather than hidden: the within-model "
    "prediction files have split assignments but no drive-index map, and "
    "the training-window subsample, the random strata of the prevalence "
    "control and the bootstrap draws are not listed but are derived from "
    "the recorded seed by the released code.",
    None,
    f"{code('README.md')} (“Artifact index”, “Known limitations of the "
    f"release”); {code('manifests/README.md')}.")

story += item(
    "B4",
    "Table 1: “Dataset” is ambiguous; rename it and keep the verification "
    "caveat. Make the distinction explicit in the repository README.",
    "Renamed. P: recomputed from the released per-window predictions alone, "
    "with the released scripts. R: a released result file whose "
    "regeneration needs a full rerun on the public dataset with the "
    "released manifest and seed, “which has not been independently "
    "re-verified end-to-end beyond the sample and split checks of Sect. 4.” "
    "The README uses the same definitions.",
    "Table 1 and caption.",
    f"{code('README.md')}.")

story += item(
    "B5",
    "Ensure every claim of “restored” coverage states same-fleet "
    "calibration, the three vendors represented, exchangeability "
    "assumptions, and that nothing is guaranteed for an unseen vendor; "
    "describe 0.330→0.828 as observed window-level coverage achieved by "
    "larger sets.",
    "Done at every place the repair is stated: abstract (“With calibration "
    "drawn from all three vendors … observed window-level failure coverage "
    "… this is not a guarantee and implies nothing for an unseen vendor”), "
    "contribution 3, Sect. 5.4 (A13), Fig. 4 caption and conclusion.",
    "Abstract; Sect. 1; Sect. 5.4; Fig. 4; Sect. 6.")

story += item(
    "B6",
    "Reference [16]/[17] numbering.",
    "See A18 and A19.",
    "References [16], [17].")

story += item(
    "B7",
    "Problem found during this audit (not raised by the reviewer).",
    "While tracing every released file we found that the stage-conditional "
    "Mondrian rows of the prevalence-control run were invalid by "
    "construction: they were calibrated on wear stages but applied to "
    "random strata (the symptom was marginal coverage near 0.44 in one "
    "stratum). No manuscript result used those rows; the manuscript uses "
    "only the split-conformal rows of that run. The code now refuses this "
    "combination, the invalid rows were removed from the released file, "
    "the full test suite passes, and the Threats paragraph states it.",
    "Sect. 6, Threats.",
    f"{code('notebooks/validate_wear.py')} ({code('_report(methods=...)')}"
    f"), {code('v5.csv')}, {code('paper_results/final/README.md')}.")

# ---------------------------------------------------------------- Part C
story.append(P("Part C. What was not done", h1))
story += item(
    "C1",
    "Independent end-to-end regeneration of results marked R.",
    "Not done. The results marked R (Fig. 2, ten-seed spread, robustness "
    "checks, held-out-vendor results, transfer diagnostics) come from the "
    "single batch run on the corrected sampler and are released as result "
    "files. Their samples and splits are checked (fingerprints, "
    "cross-session identity, consistency with the released predictions, "
    "the stage hold-out counts), but nobody has rerun the full pipeline "
    "independently and compared the outputs number by number. The "
    "manuscript says so in the Table 1 caption and the README says so in "
    "the artifact index. Results marked P can be checked by anyone from "
    "the released predictions without the raw dataset.",
    "Table 1 caption.")

# ---------------------------------------------------------------- index
story.append(P("Where to check each result", h1))
story.append(P(f"Paths relative to {code('paper_results/final/')}. "
               "Commands to regenerate are in the repository README."))
story.append(grid([
    ["Result", "Released file(s)", "Manifest: column", "Seed", "Status"],
    ["Tables 2, 3, 5, 7; Fig. 4", "preds_r2p7_*.parquet, boot_pooled.csv",
     "drives_r2p7: part_standard", "42", "P"],
    ["Table 4 (ratios)", "preds_{r1,r2p7,r5}_*.parquet",
     "drives_{r1,r2p7,r5}: part_standard", "42", "P"],
    ["Table 6, Fig. 3", "preds_wv*.parquet, boot_wv.csv",
     "drives_r2p7: part_within_vendor", "42", "P"],
    ["Within-model audit", "preds_wm*.parquet",
     "drives_r2p7: part_within_model", "42", "P"],
    ["Drive-level check", "preds_r2p7_*.parquet",
     "drives_r2p7: part_standard", "42", "P"],
    ["Fig. 2", "age_overlap.csv", "drives_r2p7: part_standard", "42", "R"],
    ["Ten-seed spread", "seeds.csv", "drives_seed0..8, drives_r2p7",
     "0–8, 42", "R"],
    ["Stage hold-out", "v3.csv",
     "drives_r2p7: modal_stage, part_holdout_stage_*", "42", "R"],
    ["Binning; classifiers; prevalence", "v2.csv; v6.csv; v5.csv",
     "drives_r2p7: part_standard", "42", "R"],
    ["Held-out vendor", "shift_rank.csv, shift_cells/",
     "drives_r2p7: part_holdout_vendor_*", "42", "R"],
    ["Representation sweep", "repr_sweep.csv, lomm_sweep.csv",
     "drives_r2p7", "42", "R"],
    ["Transfer diagnostics", "manifests/transfer_*.csv", "drives_r2p7",
     "42", "R"],
], [36, 48, 50, 18, 16]))
story.append(Spacer(1, 8))
story.append(P("Supporting files for the whole revision: "
               f"{code('manifests/samples.json')}, "
               f"{code('manifests/experiments.json')}, "
               f"{code('manifests/consistency_check.csv')}, "
               f"{code('manifests/cross_session_check.csv')}, "
               f"{code('pip_freeze.txt')}, "
               f"{code('paper/number_audit.py')}."))


def footer(c, d):
    c.saveState()
    c.setFont("DV", 7.5)
    c.setFillColor(GREY)
    c.drawString(19 * mm, 10 * mm, "Response to Reviewers · Singh, Ananshya, "
                 "Ruby D")
    c.drawRightString(A4[0] - 19 * mm, 10 * mm, f"{d.page}")
    c.restoreState()


import sys
out = sys.argv[1] if len(sys.argv) > 1 else "response_to_reviewers.pdf"
doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=19 * mm,
                        rightMargin=19 * mm, topMargin=17 * mm,
                        bottomMargin=17 * mm,
                        title="Response to Reviewers",
                        author="Aditya Aryan Singh, Ananshya, Ruby D")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", out)
