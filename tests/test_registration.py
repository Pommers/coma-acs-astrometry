"""Synthetic regression tests for ACS reconciliation + JWST registration.

Run from project root: python -m pytest -q tests/test_registration.py
"""
import numpy as np
import pandas as pd
import pytest
pytest.importorskip("astropy")

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from src.acs_astrometry import (
    Settings, reconcile_acs_catalog, find_registration_candidates,
    fit_jwst_translations, apply_jwst_translations,
    match_registered_catalogs, save_jwst_registration,
)


def test_end_to_end(tmp_path):
    rng = np.random.default_rng(2026)
    acs_parts = []
    jwst_parts = []
    images = {}
    expected = {"2-0": (-0.122, +0.435), "3-0": (+0.074, +0.355)}
    for pidx, pointing in enumerate(expected):
        n = 120
        w = WCS(naxis=2)
        w.wcs.crpix = [500, 500]
        w.wcs.cd = np.array([[-0.05 / 3600, 0], [0, 0.05 / 3600]])
        w.wcs.crval = [195.2 + pidx * 0.12, 28.0]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        image = tmp_path / f"{pointing}.fits"
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(
            name="SCI", data=np.zeros((1000, 1000), dtype="float32"),
            header=w.to_header())]).writeto(image)
        images[pointing] = image
        x = rng.uniform(100, 900, n)
        y = rng.uniform(100, 900, n)
        dx = 4.3 if pidx == 0 else 6.1
        dy = -9.24 if pidx == 0 else -2.34
        ra, dec = w.pixel_to_world_values(x+dx+rng.normal(0, 0.10, n),
                                           y+dy+rng.normal(0, 0.10, n))
        # Reproduce one inconsistent ACS row per pointing.
        ra[0], dec[0] = w.pixel_to_world_values(x[0]+25, y[0]-30)
        a = pd.DataFrame({"source_id": np.arange(n)+pidx*n,
                          "pointing": pointing, "x": x, "y": y,
                          "x_wcs": ra, "y_wcs": dec,
                          "type": ["gc", "ucd", "cand"]*(n//3)})
        acs_parts.append(a)
        good = SkyCoord(ra[1:]*u.deg, dec[1:]*u.deg)
        dra, ddec = expected[pointing]
        coords = good.spherical_offsets_by(dra*u.arcsec, ddec*u.arcsec)
        # A little centroid noise, independent of ACS positions.
        coords = coords.spherical_offsets_by(
            rng.normal(0, .008, n-1)*u.arcsec,
            rng.normal(0, .008, n-1)*u.arcsec,
        )
        j = pd.DataFrame({"source_id": np.arange(n-1)+pidx*n,
                          "ra_deg": coords.ra.deg, "dec_deg": coords.dec.deg,
                          "x0": rng.uniform(100, 900, n-1),
                          "y0": rng.uniform(100, 900, n-1),
                          "flag": 0, "peak_over_local_rms": 10.0})
        jwst_parts.append(j)
    acs = pd.concat(acs_parts, ignore_index=True)
    jwst = pd.concat(jwst_parts, ignore_index=True)
    reconciled, summary = reconcile_acs_catalog(acs, images)
    assert (summary.n_outlier == 1).all()
    assert (reconciled.type == acs.type).all()
    assert reconciled.loc[reconciled.astrom_status.eq('outlier'), 'ra_reg'].isna().all()
    candidates = find_registration_candidates(reconciled, jwst, initial_radius_arcsec=1.5)
    selected, solutions = fit_jwst_translations(candidates, min_anchors=50)
    assert (solutions.status == 'fitted').all(), solutions
    assert selected.registration_anchor.sum() > 190
    for _, row in solutions.iterrows():
        truth = expected[row.pointing]
        assert abs(row.dra_arcsec - truth[0]) < .005
        assert abs(row.ddec_arcsec - truth[1]) < .005
        assert row.p95_resid_arcsec < .04
    registered = apply_jwst_translations(reconciled, solutions)
    assert registered[['x_wcs','y_wcs','x','y','type']].equals(acs[['x_wcs','y_wcs','x','y','type']])
    assert registered.loc[registered.astrom_status.eq('outlier'), 'ra_jwst_frame'].isna().all()
    assert registered.ra_jwst_frame.notna().sum() == 238
    matched = match_registered_catalogs(registered, jwst, max_sep_arcsec=.10)
    assert len(matched) >= 225
    assert matched.sep_arcsec.median() < .02
    assert (matched.match_quality == 'high_confidence').all()
    assert matched.acs_index.is_unique
    assert matched.jwst_source_id.is_unique
    paths = save_jwst_registration(registered, solutions, matched, tmp_path / 'out',
                                   metadata={'jwst_detection_file': 'synthetic.csv'})
    assert all(path.exists() for path in paths)


def test_failed_solution_does_not_shift():
    acs = pd.DataFrame({"pointing": ["2-0"], "astrom_status": ["consistent"],
                        "ra_reg": [195.0], "dec_reg": [28.0], "type": ["cand"]})
    bad = pd.DataFrame([{"pointing":"2-0", "status":"failed",
                         "dra_arcsec": 1.0, "ddec_arcsec": 1.0}])
    result = apply_jwst_translations(acs, bad)
    assert np.isnan(result.loc[0, "ra_jwst_frame"])
    assert result.loc[0, 'type'] == 'cand'
