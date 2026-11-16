# Coma ACS Astrometry

Reproducible astrometric reconciliation and cross-instrument registration for compact stellar systems in the Coma cluster. This project audits the **existing HST/ACS parent catalogue** against its source-image WCS and, where overlapping data exist, derives pointing-specific translations to independently detected **JWST/NIRCam** sources.

The project is a self-contained *processing workflow*, not a replacement for the upstream ACS or JWST reduction pipelines. Original FITS images and SEP detection catalogues remain in their respective data projects. Their paths are supplied to the workflow; the input products are never modified.

## Current status

**Validated pilot: ACS pointings `2-0` and `3-0`, matched against the IC 4051 / PGC 44832 F150W SEP catalogue.** The clean pilot notebook reproduces the three reference CSV files from the earlier investigation exactly (matching shapes, columns, and Pandas values).

| Result | `2-0` | `3-0` | Total |
| --- | ---: | ---: | ---: |
| ACS parent records | 1,422 | 625 | 2,047 |
| Internally consistent ACS records | 1,414 | 624 | 2,038 |
| Flagged ACS-coordinate outliers | 8 | 1 | 9 |
| ACS–JWST registration anchors | 598 | 272 | 870 |
| Final reciprocal matches within 0.10″ | 839 | 308 | 1,147 |
| Matches above 0.05″, for review | 10 | 0 | 10 |

Pilot pointing translations (JWST minus ACS, on the sky):

| Pointing | ΔRA cos Dec (arcsec) | ΔDec (arcsec) |
| --- | ---: | ---: |
| `2-0` | −0.122347 | +0.435440 |
| `3-0` | +0.073846 | +0.355353 |

These values describe **this pilot and these input versions**, not global corrections for the Coma mosaic. Median final matched separations are approximately 0.0125″ (`2-0`) and 0.0113″ (`3-0`). The translation-fit residuals are **in-sample**, not an independent accuracy estimate. Reciprocal associations and the `high_confidence` label remain provisional pending source-identity and crowding checks.

## Repository layout

```text
coma_acs_astrometry/
├── README.md
├── src/
│   ├── __init__.py
│   └── acs_astrometry.py
├── tests/
│   ├── test_smoke.py
│   └── test_registration.py
├── notebooks/
│   └── 01_astrometry_pilot.ipynb
├── data/
│   └── coma_astrometry/
│       └── processed/      # generated, not committed
└── figures/                # generated, not committed
```

File names shown for the notebook and smoke test are conventional; use the actual filenames present in your checkout. The algorithms live in `src/`, not in notebooks.

## Dependencies and tests

Run in the existing astronomy Python environment (used for the pilot: `jwst_phot`). Core dependencies are `numpy`, `pandas`, `astropy`, and `pytest` for tests; notebooks and diagnostic figures additionally use `jupyter` and `matplotlib`.

From the repository root:

```bash
python -m pytest -q tests/test_registration.py
# Run the existing smoke test using its established invocation.
```

The synthetic regression tests passed locally. The clean, real-data pilot also produced CSV outputs **exactly equal** to its three previously saved reference products:

```text
acs_registered_2-0_3-0.csv          2,047 rows × 25 columns
acs_jwst_registration_2-0_3-0.csv     2 rows × 6 columns
acs_jwst_matches_2-0_3-0.csv      1,147 rows × 29 columns
```

A warning from AnyIO's pytest plugin about `CallSpec2` is a third-party deprecation warning and did not cause the tests to fail.

## Portable paths and external input data

The notebook should discover **its own repository root** rather than hard-code `/Users/<username>/PycharmProjects`. For the usual sibling-project layout, `HOME_DIR` is simply `PROJECT_ROOT.parent`.

Copy the bootstrap block from [`notebook_paths.py`](notebook_paths.py) into the first code cell of the notebook, or execute that file with `%run` if it is kept at the repository root. The bootstrap searches upward from the current working directory for `src/acs_astrometry.py`, so it works from the project root **or** its `notebooks/` directory.

By default it assumes:

```text
<workspace>/
├── coma_acs_astrometry/
├── Coma_Intracluster_GC/
│   ├── regions/           # ACS FITS
│   └── data/              # original ACS catalogue
└── <separate JWST project>/ # JWST imaging and SEP source catalogue
```

To use a different external ACS project location without changing committed code, set `COMA_ACS_PROJECT_DIR` before starting Jupyter (or in your IDE run configuration):

```bash
export COMA_ACS_PROJECT_DIR="/path/to/Coma_Intracluster_GC"
```

An optional `COMA_WORKSPACE_DIR` overrides the default parent directory; `COMA_JWST_PROJECT_DIR` can identify the separate JWST project when needed. Paths are **local configuration**, not something to commit to Git. The bootstrap does not change the working directory.

Example for the two ACS FITS images:

```python
pilot_images = {
    "2-0": REGIONS_DIR / "2-0/jc1g01010_drc.fits",
    "3-0": REGIONS_DIR / "3-0/jc1g02010_drc.fits",
}
```

The independently detected JWST input for the current pilot is `pgc44832_F150W_SEP_detections_v01.csv` (4,740 SEP detections). Its **actual location** must be configured in the notebook; the astrometry package does not generate this input and does not infer its location.

## Processing workflow

1. **Load the unfiltered ACS parent catalogue** and explicitly select pilot pointings. Preserve the original upstream `type` values (`gc`, `ucd`, `cand`), catalogue sky positions, image pixels, and photometry.
2. **Reconcile each ACS pointing:** `reconcile_acs_catalog()` compares the stored pixel positions and sky coordinates against that pointing's own ACS image WCS, fits the bulk coordinate relationship and retains/flags anomalous records. Outlier `ra_reg`/`dec_reg` values are missing to prevent automatic matching.
3. **Find potential cross-instrument associations:** `find_registration_candidates()` uses independent F150W SEP positions, quality cuts, reciprocal nearest neighbors, and an exploratory 1.5″ search radius.
4. **Fit pointing-specific ACS→JWST shifts:** `fit_jwst_translations()` selects the compact offset populations (pilot anchor radius 0.08″) and estimates a translation separately for `2-0` and `3-0`.
5. **Apply and match:** `apply_jwst_translations()` creates new JWST-frame coordinate columns, preserving the ACS originals; `match_registered_catalogs()` performs reciprocal matching within 0.10″ and flags separations above 0.05″ for review.
6. **Export:** save the augmented parent catalogue, translation table, and match table to `data/coma_astrometry/processed/`, with a record of input filenames and thresholds. Compare against the known-good pilot outputs when modifying the helper.

Main helper API:

```python
from src.acs_astrometry import (
    Settings,
    reconcile_acs_catalog,
    find_registration_candidates,
    fit_jwst_translations,
    apply_jwst_translations,
    match_registered_catalogs,
    save_reconciliation,
    save_jwst_registration,
)
```

**Coordinate convention:** the ACS parent catalogue currently calls its RA/Dec columns `x_wcs`/`y_wcs`; these are *degrees*, not image pixels. The true ACS pixel columns are `x`/`y`. SEP's `x0`/`y0` are zero-based pixels, and `ra_deg`/`dec_deg` are sky coordinates. Ensure image product, WCS extension and pixel origin correspond before using new ACS pointings.

## Scientific/data-integrity safeguards

- **Do not overwrite original ACS coordinates, photometry or `type` classifications.** Keep all parent records, including unmatched sources and flagged astrometric outliers.
- The bulk ACS-to-image WCS offset is a difference between coordinate solutions; it is not by itself proof that the original catalogue RA/Dec is wrong.
- Apply **per-pointing** corrections only where supported by independent JWST associations. Do not extrapolate the `2-0`/`3-0` solutions to other tiles or filters.
- Registration anchors are not a blind validation sample. Independent or withheld-source checks are needed for a formal estimate of absolute astrometric accuracy.
- Registration is distinct from photometric matching. A small positional separation alone does not verify a physical association in a crowded or high-background field.
- Mixed ACS `_drc`, `_drz` and custom mosaic FITS products may have different SCI extensions, WCS and pixel-coordinate provenance. Inspect them before processing.
- The current goal is a reusable **ACS-coordinate audit and cross-instrument registration component**. Source detection, galaxy subtraction, forced photometry, multi-filter matching and final catalogue science remain in their respective pipelines.

## Next steps

1. Inventory the remaining ACS FITS products and confirm image/HDU/WCS provenance.
2. Run ACS-only reconciliation for every pointing; save a mosaic-wide quality summary and flag exceptions.
3. Register the overlapping pointings against appropriate independent JWST data by filter/field, without mixing cross-instrument provenance.
4. Feed validated coordinates and quality flags to downstream JWST matching and optical–NIR photometry.

**Operating principle: _No New Rabbit-holes._** Quantify, flag, document, and proceed unless a discrepancy affects science or blocks a reproducible result.
