# ACS astrometric reconciliation

Stand-alone helper for the Coma ACS–JWST catalog. The input ACS parent catalog is never overwritten; all original `gc` / `ucd` / `cand` `type` values survive untouched.

## Dependencies

```bash
pip install numpy pandas astropy
```

## Notebook usage

```python
import pandas as pd
from acs_astrometry import Settings, reconcile_acs_catalog, validate_with_jwst, save_reconciliation

cfg = Settings(ra_col="x_wcs", dec_col="y_wcs", pixel_origin=0)
acs = pd.read_csv("path/to/unfiltered_acs_parent.csv")
images = {
    "2-0": "regions/2-0/jc1g01010_drc.fits",
    "3-0": "regions/3-0/jc1g02010_drc.fits",
    # Add all pointings represented in the input catalog.
}

checked, pointings = reconcile_acs_catalog(acs, images, cfg)
display(pointings)
display(checked[checked.astrom_status == "outlier"])

# Optional: independently measured and pre-vetted JWST source pairs.
# Columns: source_id, jwst_ra, jwst_dec
# Do NOT generate these anchor pairs solely by nearest-neighbor matching to
# the same ACS coordinates you are trying to validate.
anchors = pd.read_csv("path/to/high_confidence_independent_jwst_anchors.csv")
checked, pointings = validate_with_jwst(
    checked, pointings, anchors, cfg, id_col="source_id",
    apply_validated_shift=False,
)

save_reconciliation(checked, pointings, "outputs/astrometry", cfg)
```

## Science cautions

- The bulk offset `dx_bulk_pix`, `dy_bulk_pix` is **not** automatically applied as a sky correction. It relates the old ACS image WCS to the catalog's own sky positions; it may be intentional.
- The internal-consistency flag identifies records that do not follow the bulk pixel–sky transformation, not which coordinate set is right.
- Only a pre-vetted independent JWST reference can validate the absolute positions. Optional `apply_validated_shift=True` adjusts **only** internally consistent ACS entries after passing a robust JWST check, so use this option only if an absolute correction is warranted.
- Records flagged `outlier` retain original positions/classifications but get NaN in `ra_reg`/`dec_reg` to prevent unintended automated photometry and crossmatching.
- `min_jwst_anchors` defaults to 10. No JWST coverage means `consistent` (not `validated`); do not interpret lack of coverage as bad astrometry.
- Before downstream use, verify that FITS images and stored pixel coordinates share the same pixel origin and image-product provenance.
- For an ACS catalog with multiple rows per source across tiles, create a unique composite source ID before using the JWST anchor stage.
