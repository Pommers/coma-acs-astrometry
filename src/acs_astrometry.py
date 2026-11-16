"""Conservative, reproducible ACS astrometric reconciliation.

The ACS catalogue's celestial positions are NOT assumed to be wrong merely
because they differ from the image-header WCS.  A robust per-pointing offset
between WCS-projected catalogue RA/Dec and stored pixel X/Y is estimated for
quality control, not automatically applied as an absolute correction.

Optional JWST validation takes independently measured, pre-vetted matches.
It estimates a small tangent-plane correction for a pointing if enough anchors
are supplied; it does not create or validate its own source associations.

Requires: numpy, pandas, astropy. Designed for zero-indexed image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping, Optional
import json

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u


@dataclass(frozen=True)
class Settings:
    pointing_col: str = "pointing"
    x_col: str = "x"
    y_col: str = "y"
    ra_col: str = "x_wcs"  # user's catalogue column holding RA, in degrees
    dec_col: str = "y_wcs" # user's catalogue column holding Dec, in degrees
    type_col: str = "type"
    sci_ext: str | int = "SCI"
    pixel_origin: int = 0  # input pixels are NumPy (0) or FITS (1) indexed
    max_internal_residual_pix: float = 1.0
    min_internal_objects: int = 15
    min_jwst_anchors: int = 10
    max_jwst_scatter_arcsec: float = 0.10
    jwst_clip_sigma: float = 4.0
    jwst_clip_floor_arcsec: float = 0.015
    jwst_max_iterations: int = 8


def _angular_offsets_arcsec(ref_ra_deg, ref_dec_deg, target_ra_deg, target_dec_deg):
    """Target minus reference in the local tangent plane (east, north), arcsec."""
    ref = SkyCoord(np.asarray(ref_ra_deg)*u.deg, np.asarray(ref_dec_deg)*u.deg, frame="icrs")
    target = SkyCoord(np.asarray(target_ra_deg)*u.deg, np.asarray(target_dec_deg)*u.deg, frame="icrs")
    dlon, dlat = ref.spherical_offsets_to(target)
    return np.asarray(dlon.arcsec), np.asarray(dlat.arcsec)


def _robust_center_xy(dx, dy, max_iter=8, clip_sigma=4.0, floor=0.15):
    """Median 2D displacement; clip radial deviations, with an explicit floor."""
    dx = np.asarray(dx, float)
    dy = np.asarray(dy, float)
    good = np.isfinite(dx) & np.isfinite(dy)
    if not np.any(good):
        raise ValueError("No finite offsets")
    inlier = good.copy()
    for _ in range(max_iter):
        mx, my = np.median(dx[inlier]), np.median(dy[inlier])
        r = np.hypot(dx-mx, dy-my)
        rm = np.median(r[inlier])
        sigma = 1.4826*np.median(np.abs(r[inlier]-rm))
        cut = max(floor, clip_sigma*sigma)
        updated = good & (r <= cut)
        if not np.any(updated) or np.array_equal(updated, inlier):
            break
        inlier = updated
    return float(np.median(dx[inlier])), float(np.median(dy[inlier])), inlier


def _wcs_from_fits(filename, sci_ext):
    with fits.open(filename, memmap=True) as hdul:
        hdu = hdul[sci_ext]
        if hdu.data is None or hdu.data.ndim != 2:
            raise ValueError(f"SCI must have a 2D image: {filename}")
        # Passing hdul also allows access to distortion lookup extensions.
        wcs = WCS(hdu.header, hdul).celestial
        return wcs, tuple(hdu.data.shape)


def reconcile_acs_catalog(
    catalog: pd.DataFrame,
    pointing_fits: Mapping[str, str | Path],
    settings: Settings = Settings(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit internal ACS pixel/sky consistency, pointing by pointing.

    Output is a COPY of the original catalog, with new `astrom_*` and `ra_reg`,
    `dec_reg` columns.  Original `x`, `y`, sky and `type` columns are untouched.

    `ra_reg`, `dec_reg` initially equal the original catalogue sky positions,
    only for internal-consistent objects. Outliers get NaN registered positions
    so downstream matching can't accidentally rely on them.  These are not
    independently JWST validated until validate_with_jwst() has been run.
    """
    required = [settings.pointing_col, settings.x_col, settings.y_col,
                settings.ra_col, settings.dec_col]
    absent = [c for c in required if c not in catalog]
    if absent:
        raise ValueError(f"Missing columns: {absent}")
    if settings.pixel_origin not in (0, 1):
        raise ValueError("pixel_origin must be 0 or 1")
    if not catalog.index.is_unique:
        raise ValueError("Catalogue index must be unique: use a stable source ID index or reset_index")

    result = catalog.copy(deep=True)
    for col in ("astrom_dx_pix", "astrom_dy_pix", "astrom_resid_pix",
                "ra_reg", "dec_reg"):
        result[col] = np.nan
    result["astrom_status"] = "unvalidated"
    result["astrom_ref"] = "acs_catalog_original"
    summaries = []

    for pointing, indices in result.groupby(settings.pointing_col, sort=False).groups.items():
        path = pointing_fits.get(str(pointing))
        if path is None:
            raise KeyError(f"Missing image mapping for pointing {pointing!r}")
        wcs, shape = _wcs_from_fits(path, settings.sci_ext)
        subset = result.loc[indices]
        x = subset[settings.x_col].to_numpy(float) - settings.pixel_origin
        y = subset[settings.y_col].to_numpy(float) - settings.pixel_origin
        ra = subset[settings.ra_col].to_numpy(float)
        dec = subset[settings.dec_col].to_numpy(float)
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(ra) & np.isfinite(dec)
        # Use NaN placeholders so any invalid rows are flagged rather than dropped.
        xp, yp = np.full(len(subset), np.nan), np.full(len(subset), np.nan)
        if finite.any():
            xp[finite], yp[finite] = wcs.world_to_pixel_values(ra[finite], dec[finite])
        dx, dy = xp-x, yp-y
        if finite.sum() < settings.min_internal_objects:
            raise ValueError(f"{pointing}: only {finite.sum()} valid objects; need at least {settings.min_internal_objects}")
        mx, my, fit_mask = _robust_center_xy(dx, dy)
        resid = np.hypot(dx-mx, dy-my)
        # Coordinates outside the SCI frame are not considered usable.
        ny, nx = shape
        inside = finite & (x >= 0) & (y >= 0) & (x < nx) & (y < ny)
        good = inside & np.isfinite(resid) & (resid <= settings.max_internal_residual_pix)

        result.loc[indices, "astrom_dx_pix"] = dx
        result.loc[indices, "astrom_dy_pix"] = dy
        result.loc[indices, "astrom_resid_pix"] = resid
        result.loc[indices, "astrom_status"] = np.where(good, "consistent", "outlier")
        result.loc[indices, "ra_reg"] = np.where(good, ra, np.nan)
        result.loc[indices, "dec_reg"] = np.where(good, dec, np.nan)
        summaries.append({
            "pointing": str(pointing), "fits_file": str(Path(path).resolve()),
            "sci_ext": str(settings.sci_ext), "n_total": len(subset),
            "n_fit": int(fit_mask.sum()), "n_consistent": int(good.sum()),
            "n_outlier": int((~good).sum()),
            "dx_bulk_pix": mx, "dy_bulk_pix": my,
            "median_residual_pix": float(np.median(resid[good])) if good.any() else np.nan,
            "p95_residual_pix": float(np.percentile(resid[good], 95)) if good.any() else np.nan,
            "jwst_status": "not_checked", "jwst_dx_arcsec": np.nan,
            "jwst_dy_arcsec": np.nan, "jwst_scatter_arcsec": np.nan,
            "jwst_n_anchors": 0,
        })
    return result, pd.DataFrame(summaries)


def validate_with_jwst(
    reconciled: pd.DataFrame,
    summary: pd.DataFrame,
    anchors: pd.DataFrame,
    settings: Settings = Settings(),
    id_col: str = "source_id",
    jwst_ra_col: str = "jwst_ra",
    jwst_dec_col: str = "jwst_dec",
    apply_validated_shift: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the ACS sky frame with independently detected JWST sources.

    `anchors` contains vetted, unique (ACS source ID, JWST centroid) pairs;
    NO crossmatching is performed here.  The catalogue must already have an
    `id_col` unique identifier, shared with anchors.  If the robust JWST offset
    scatter passes threshold, the pointing is labelled `validated`.

    An absolute shift is optional and **off by default**; use it only when the
    shift is scientifically justified, e.g. the independent JWST reference frame
    has verified absolute calibration.  Store that decision in metadata.
    """
    needed = [id_col, jwst_ra_col, jwst_dec_col]
    if any(c not in anchors for c in needed):
        raise ValueError(f"anchors must include {needed}")
    if id_col not in reconciled:
        raise ValueError(f"reconciled catalogue must include {id_col!r}")
    if anchors[id_col].duplicated().any():
        raise ValueError("JWST anchor IDs must be unique; resolve ambiguous matches first")
    if reconciled[id_col].duplicated().any():
        raise ValueError("ACS source IDs must be unique; use a composite unique ID if needed")

    result = reconciled.copy(deep=True)
    summaries = summary.copy(deep=True)
    merged = result[[id_col, settings.pointing_col, "ra_reg", "dec_reg", "astrom_status"]].merge(
        anchors[needed], on=id_col, how="inner", validate="one_to_one"
    )
    merged = merged[merged["astrom_status"] == "consistent"].copy()
    for pointing, section in merged.groupby(settings.pointing_col):
        if len(section) < settings.min_jwst_anchors:
            continue
        dra, ddec = _angular_offsets_arcsec(
            section["ra_reg"].to_numpy(), section["dec_reg"].to_numpy(),
            section[jwst_ra_col].to_numpy(), section[jwst_dec_col].to_numpy()
        )
        dx, dy, mask = _robust_center_xy(
            dra, ddec, max_iter=settings.jwst_max_iterations,
            clip_sigma=settings.jwst_clip_sigma,
            floor=settings.jwst_clip_floor_arcsec
        )
        rr = np.hypot(dra[mask]-dx, ddec[mask]-dy)
        med_scatter = float(np.median(rr))
        verified = mask.sum() >= settings.min_jwst_anchors and med_scatter <= settings.max_jwst_scatter_arcsec
        loc = summaries["pointing"].astype(str) == str(pointing)
        summaries.loc[loc, "jwst_status"] = "validated" if verified else "failed"
        summaries.loc[loc, "jwst_dx_arcsec"] = dx
        summaries.loc[loc, "jwst_dy_arcsec"] = dy
        summaries.loc[loc, "jwst_scatter_arcsec"] = med_scatter
        summaries.loc[loc, "jwst_n_anchors"] = int(mask.sum())
        if verified:
            valid_rows = (result[settings.pointing_col] == pointing) & (result["astrom_status"] == "consistent")
            result.loc[valid_rows, "astrom_status"] = "validated"
            if apply_validated_shift:
                # Spherical offset to keep angular geometry correct.
                coords = SkyCoord(result.loc[valid_rows, "ra_reg"].to_numpy()*u.deg,
                                  result.loc[valid_rows, "dec_reg"].to_numpy()*u.deg)
                shifted = coords.spherical_offsets_by(dx*u.arcsec, dy*u.arcsec)
                result.loc[valid_rows, "ra_reg"] = shifted.ra.deg
                result.loc[valid_rows, "dec_reg"] = shifted.dec.deg
                result.loc[valid_rows, "astrom_ref"] = "jwst_shifted"
            else:
                result.loc[valid_rows, "astrom_ref"] = "acs_catalog_jwst_checked"
    return result, summaries


def save_reconciliation(
    catalog: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path,
    settings: Settings = Settings(),
    prefix: str = "acs_astrometry",
):
    """Persist products and explicit input/configuration metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cat_path = output_dir / f"{prefix}_catalog.csv"
    sum_path = output_dir / f"{prefix}_pointings.csv"
    cfg_path = output_dir / f"{prefix}_settings.json"
    catalog.to_csv(cat_path, index=True, index_label="input_row_index")
    summary.to_csv(sum_path, index=False)
    with cfg_path.open("w") as f:
        json.dump(asdict(settings), f, indent=2)
    return cat_path, sum_path, cfg_path

# -----------------------------------------------------------------------------
# ACS <-> JWST cross-instrument registration (independent SEP detections)
# -----------------------------------------------------------------------------

def _jwst_quality_subset(
    detections: pd.DataFrame,
    flag_col: str = "flag",
    snr_col: str = "peak_over_local_rms",
    min_snr: float = 5.0,
    min_edge_px: float = 20.0,
    image_shape: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Choose initial SEP registration/matching candidates; never change the input.

    x0/y0 are zero-based SEP positions. With image_shape=(ny,nx), both ends of
    each image axis are checked; without it only the lower boundaries are cut.
    """
    needed = ["ra_deg", "dec_deg", "x0", "y0", flag_col, snr_col, "source_id"]
    missing = [c for c in needed if c not in detections.columns]
    if missing:
        raise ValueError(f"Missing JWST detection columns: {missing}")
    if detections["source_id"].duplicated().any():
        raise ValueError("JWST source_id must be unique")
    good = (
        (detections[flag_col] == 0)
        & (detections[snr_col] >= min_snr)
        & np.isfinite(detections["ra_deg"])
        & np.isfinite(detections["dec_deg"])
        & (detections["x0"] > min_edge_px)
        & (detections["y0"] > min_edge_px)
    )
    if image_shape is not None:
        ny, nx = image_shape
        good &= (detections["x0"] < nx - 1 - min_edge_px) & (detections["y0"] < ny - 1 - min_edge_px)
    return detections.loc[good].copy()


def _reciprocal_pairs(
    acs: pd.DataFrame,
    jwst: pd.DataFrame,
    ra_col: str,
    dec_col: str,
    max_sep_arcsec: float,
) -> pd.DataFrame:
    """Reciprocal nearest-neighbor sky matches; indices reference original input rows."""
    if not acs.index.is_unique or not jwst.index.is_unique:
        raise ValueError("ACS and JWST DataFrame indices must each be unique")
    if acs.empty or jwst.empty:
        return pd.DataFrame(columns=["acs_index", "jwst_index", "sep_arcsec"])
    a = SkyCoord(acs[ra_col].to_numpy(float)*u.deg, acs[dec_col].to_numpy(float)*u.deg)
    j = SkyCoord(jwst["ra_deg"].to_numpy(float)*u.deg, jwst["dec_deg"].to_numpy(float)*u.deg)
    j_for_a, sep, _ = a.match_to_catalog_sky(j)
    a_for_j, _, _ = j.match_to_catalog_sky(a)
    keep = (a_for_j[j_for_a] == np.arange(len(acs))) & (sep.arcsec < max_sep_arcsec)
    ai = np.flatnonzero(keep)
    ji = j_for_a[keep]
    return pd.DataFrame({
        "acs_index": acs.index.to_numpy()[ai],
        "jwst_index": jwst.index.to_numpy()[ji],
        "sep_arcsec": sep.arcsec[keep],
    })


def find_registration_candidates(
    reconciled: pd.DataFrame,
    jwst_detections: pd.DataFrame,
    *,
    initial_radius_arcsec: float = 1.5,
    min_snr: float = 5.0,
    min_edge_px: float = 20.0,
    jwst_image_shape: tuple[int, int] | None = None,
    settings: Settings = Settings(),
) -> pd.DataFrame:
    """Exploratory reciprocal ACS--JWST associations using original ACS sky positions.

    Candidates are NOT scientifically confirmed matches. Offsets are JWST-ACS.
    A source's native catalogue index is retained as `acs_index`.
    """
    a = reconciled.loc[
        reconciled["astrom_status"].eq("consistent")
        & reconciled["ra_reg"].notna()
        & reconciled["dec_reg"].notna()
    ].copy()
    j = _jwst_quality_subset(jwst_detections, min_snr=min_snr,
                             min_edge_px=min_edge_px, image_shape=jwst_image_shape)
    pairs = _reciprocal_pairs(a, j, "ra_reg", "dec_reg", initial_radius_arcsec)
    if pairs.empty:
        return pairs.assign(pointing=pd.Series(dtype=str), jwst_source_id=pd.Series(dtype=object),
                            dra_arcsec=pd.Series(dtype=float), ddec_arcsec=pd.Series(dtype=float))
    ac = a.loc[pairs["acs_index"]]
    js = j.loc[pairs["jwst_index"]]
    dra, ddec = _angular_offsets_arcsec(ac["ra_reg"].to_numpy(), ac["dec_reg"].to_numpy(),
                                       js["ra_deg"].to_numpy(), js["dec_deg"].to_numpy())
    pairs["pointing"] = ac[settings.pointing_col].astype(str).to_numpy()
    pairs["jwst_source_id"] = js["source_id"].to_numpy()
    pairs["dra_arcsec"] = dra
    pairs["ddec_arcsec"] = ddec
    return pairs


def fit_jwst_translations(
    candidates: pd.DataFrame,
    *,
    anchor_radius_arcsec: float = 0.08,
    min_anchors: int = 10,
    max_anchor_p95_arcsec: float = 0.08,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the dominant *translation* in each ACS pointing.

    Median-center and select offsets inside `anchor_radius_arcsec`, then recenter
    on selected anchors. Return (annotated_candidates, pointing_solutions).
    The radius selects the fitting subset; statistics are therefore IN-SAMPLE.
    A stable initial compact cluster is assumed: inspect plots before use in
    dense/ambiguous fields. A 'failed' solution must not be applied.
    """
    needed = {"pointing", "dra_arcsec", "ddec_arcsec"}
    if not needed.issubset(candidates):
        raise ValueError(f"Candidates need columns {needed}")
    if anchor_radius_arcsec <= 0:
        raise ValueError("anchor_radius_arcsec must be positive")
    annotated = candidates.copy()
    annotated["registration_resid_arcsec"] = np.nan
    annotated["registration_anchor"] = False
    rows = []
    for p, idx in annotated.groupby("pointing", sort=False).groups.items():
        g = annotated.loc[idx]
        dra = g["dra_arcsec"].to_numpy(float)
        ddec = g["ddec_arcsec"].to_numpy(float)
        finite = np.isfinite(dra) & np.isfinite(ddec)
        if not finite.any():
            rows.append({"pointing": str(p), "status": "failed", "n_candidates": len(g),
                         "n_anchors": 0, "dra_arcsec": np.nan, "ddec_arcsec": np.nan,
                         "median_resid_arcsec": np.nan, "p95_resid_arcsec": np.nan})
            continue
        cx, cy = np.median(dra[finite]), np.median(ddec[finite])
        distance = np.hypot(dra-cx, ddec-cy)
        mask = finite & (distance < anchor_radius_arcsec)
        # One refinement, matching the exploratory notebook's anchor procedure.
        if mask.any():
            cx, cy = np.median(dra[mask]), np.median(ddec[mask])
        residual = np.hypot(dra-cx, ddec-cy)
        annotated.loc[idx, "registration_resid_arcsec"] = residual
        annotated.loc[idx, "registration_anchor"] = mask
        med = float(np.median(residual[mask])) if mask.any() else np.nan
        p95 = float(np.percentile(residual[mask], 95)) if mask.any() else np.nan
        status = "fitted" if (mask.sum() >= min_anchors and p95 <= max_anchor_p95_arcsec) else "failed"
        rows.append({"pointing": str(p), "status": status, "n_candidates": len(g),
                     "n_anchors": int(mask.sum()), "dra_arcsec": float(cx), "ddec_arcsec": float(cy),
                     "median_resid_arcsec": med, "p95_resid_arcsec": p95})
    return annotated, pd.DataFrame(rows)


def apply_jwst_translations(
    reconciled: pd.DataFrame,
    solutions: pd.DataFrame,
    *,
    settings: Settings = Settings(),
) -> pd.DataFrame:
    """Add JWST-frame positions without changing ACS original or `ra_reg`/`dec_reg`.

    Only pointings with status='fitted' and internally consistent ACS objects
    receive new coordinates. Angular displacements use spherical_offsets_by.
    """
    out = reconciled.copy(deep=True)
    out["ra_jwst_frame"] = np.nan
    out["dec_jwst_frame"] = np.nan
    if solutions["pointing"].duplicated().any():
        raise ValueError("Solutions must contain one row per pointing")
    for _, s in solutions.iterrows():
        if s["status"] != "fitted":
            continue
        mask = (out[settings.pointing_col].astype(str) == str(s["pointing"])) & out["astrom_status"].eq("consistent")
        if not mask.any():
            continue
        coords = SkyCoord(out.loc[mask, "ra_reg"].to_numpy(float)*u.deg,
                          out.loc[mask, "dec_reg"].to_numpy(float)*u.deg)
        shifted = coords.spherical_offsets_by(float(s["dra_arcsec"])*u.arcsec,
                                              float(s["ddec_arcsec"])*u.arcsec)
        out.loc[mask, "ra_jwst_frame"] = shifted.ra.deg
        out.loc[mask, "dec_jwst_frame"] = shifted.dec.deg
    return out


def match_registered_catalogs(
    registered: pd.DataFrame,
    jwst_detections: pd.DataFrame,
    *,
    max_sep_arcsec: float = 0.10,
    review_above_arcsec: float = 0.05,
    min_snr: float = 5.0,
    min_edge_px: float = 20.0,
    jwst_image_shape: tuple[int, int] | None = None,
    settings: Settings = Settings(),
) -> pd.DataFrame:
    """Final exploratory reciprocal pairs at registered ACS positions.

    Output is a MATCH TABLE, not the ACS parent catalogue. `high_confidence`
    labels are positional-only and do NOT certify photometric identity.
    """
    a = registered.loc[
        registered["astrom_status"].eq("consistent")
        & registered["ra_jwst_frame"].notna()
        & registered["dec_jwst_frame"].notna()
    ]
    j = _jwst_quality_subset(jwst_detections, min_snr=min_snr,
                             min_edge_px=min_edge_px, image_shape=jwst_image_shape)
    pairs = _reciprocal_pairs(a, j, "ra_jwst_frame", "dec_jwst_frame", max_sep_arcsec)
    if pairs.empty:
        return pairs.assign(pointing=pd.Series(dtype=str), jwst_source_id=pd.Series(dtype=object),
                            match_quality=pd.Series(dtype=str))
    pairs["pointing"] = a.loc[pairs["acs_index"], settings.pointing_col].astype(str).to_numpy()
    pairs["jwst_source_id"] = j.loc[pairs["jwst_index"], "source_id"].to_numpy()
    pairs["match_quality"] = np.where(pairs["sep_arcsec"] <= review_above_arcsec,
                                       "high_confidence", "review")
    return pairs


def save_jwst_registration(
    registered: pd.DataFrame,
    solutions: pd.DataFrame,
    matches: pd.DataFrame,
    output_dir: str | Path,
    *,
    prefix: str = "acs_jwst",
    metadata: dict | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Save parent-preserving catalog, per-pointing solutions, matches and manifest."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = (directory / f"{prefix}_registered.csv",
             directory / f"{prefix}_translations.csv",
             directory / f"{prefix}_matches.csv",
             directory / f"{prefix}_manifest.json")
    registered.to_csv(paths[0], index=True, index_label="acs_catalog_index")
    solutions.to_csv(paths[1], index=False)
    matches.to_csv(paths[2], index=False)
    with paths[3].open("w") as f:
        json.dump(metadata or {}, f, indent=2, default=str)
    return paths
