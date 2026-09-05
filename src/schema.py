"""
SMART column schema.

ASSUMED, NOT VERIFIED. The real Alibaba parquet has 105 columns:
3 metadata (disk_id, ds, model) + 102 SMART = 51 SMART IDs x {n_i, r_i}.
The IDs below are a plausible SSD-relevant set chosen to hit that count.

TO REPLACE WITH THE REAL SCHEMA:
    import polars as pl
    print(pl.scan_parquet(PARQUET).collect_schema().names())
then overwrite SMART_IDS (or SMART_COLS directly) here. Nothing else in
the codebase should hardcode column names.
"""

from .config import META_COLS

# 51 SMART IDs. Standard ATA attributes plus the vendor-specific 17x/18x
# block that SSDs use for wear, program/erase failures and reserved blocks.
SMART_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    170, 171, 172, 173, 174, 177,
    180, 181, 182, 183, 184,
    187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
    204, 205, 206, 207, 211,
    233, 240, 241, 242, 244, 245,
    175, 232
]

NORM_COLS = [f"n_{i}" for i in SMART_IDS]
RAW_COLS = [f"r_{i}" for i in SMART_IDS]

# Interleaved n_i, r_i — matches the readme's field ordering.
SMART_COLS = [c for i in SMART_IDS for c in (f"n_{i}", f"r_{i}")]

ALL_COLS = META_COLS + SMART_COLS

assert len(SMART_IDS) == 51, len(SMART_IDS)
assert len(ALL_COLS) == 105, len(ALL_COLS)


# ---------------------------------------------------------------
# Per-vendor attribute availability
#
# Vendors populate different SMART ID subsets. This divergence IS the
# covariate shift the paper measures, so the synthetic generator must
# reproduce it. Values below are invented; replace from
# reports/attribute_availability.csv once profiling has run.
# ---------------------------------------------------------------

_COMMON = [1, 5, 9, 12, 187, 190, 194, 195, 197, 198, 199, 241, 242]
_COMMON = [i for i in _COMMON if i in SMART_IDS]

VENDOR_SMART_IDS = {
    "A": [
        5, 9, 12, 170, 171, 172, 174, 175, 183, 184,
        187, 190, 192, 194, 197, 199, 232, 233, 241, 242
    ],

    "B": [
        5, 9, 12, 177, 180, 181, 182, 183, 184,
        187, 190, 195, 197, 199, 241, 242, 244, 245
    ],

    "C": [
        1, 5, 9, 12, 170, 171, 172, 173, 174, 180,
        183, 184, 187, 188, 194, 195, 196, 197, 198, 199, 206
    ],
}

VENDOR_SMART_IDS = {
    v: sorted(set(i for i in ids if i in SMART_IDS))
    for v, ids in VENDOR_SMART_IDS.items()
}


# ---------------------------------------------------------------
# Common feature set — the LOMO-usable intersection
#
# Only 8 SMART IDs (16 columns) are populated by all three vendors.
# Under LOMO this is the only feature set available in every fold:
# columns exclusive to the training vendors are all-NaN at test time,
# and columns exclusive to the held-out vendor were never trained on.
#
# LOCKED DECISION: common-16 is the primary feature policy. A
# union-plus-imputation variant may be run as an ablation, but
# imputation must not confound the headline coverage result.
# ---------------------------------------------------------------

COMMON_IDS = sorted(
    set(VENDOR_SMART_IDS["A"])
    & set(VENDOR_SMART_IDS["B"])
    & set(VENDOR_SMART_IDS["C"])
)

COMMON_COLS = [c for i in COMMON_IDS for c in (f"n_{i}", f"r_{i}")]

assert len(COMMON_COLS) == 16, (
    f"expected 16 common columns, got {len(COMMON_COLS)}. "
    "If this changed, re-derive from reports/attribute_availability.csv "
    "and revisit the feature policy."
)


def common_cols() -> list[str]:
    """The 16 columns usable in every LOMO fold."""
    return list(COMMON_COLS)


def vendor_cols(vendor: str) -> list[str]:
    """SMART columns a vendor populates. Others are NaN throughout."""
    ids = VENDOR_SMART_IDS[vendor]
    return [c for i in ids for c in (f"n_{i}", f"r_{i}")]