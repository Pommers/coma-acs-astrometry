# logging function
import logging
import pandas as pd

def setup_logging(verbose=False, name="Coma_GCs_debug"):
    """
    Call this once at the top of the notebook.
    If verbose=True, DEBUG messages will show; otherwise only INFO+.
    """
    # name = "Coma_GCs_debug"
    
    # Remove any root handlers (if I had root-level logging before)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    # Get named logger and clear *its* handlers + disable propagation
    logger = logging.getLogger(name)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    logger.propagate = False

    # Set logger’s level
    logger.setLevel(logging.DEBUG)   # we capture everything here

    # INFO-only handler (no timestamp)
    info_handler = logging.StreamHandler()
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(lambda rec: rec.levelno == logging.INFO)
    info_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(info_handler)

    # non-INFO handler (timestamped, DEBUG or WARNING+)
    other = logging.StreamHandler()
    other.setLevel(logging.DEBUG if verbose else logging.WARNING)
    other.addFilter(lambda rec: rec.levelno != logging.INFO)
    # allow only your logger’s DEBUG (if you still want a name-filter):
    other.addFilter(lambda rec: rec.levelno != logging.DEBUG or rec.name == name)
    fmt = "[%(levelname)s] %(asctime)s.%(msecs)03d %(message)s"
    other.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logger.addHandler(other)


#--------------------------------------------------------------------------------------------#

from pathlib import Path
import yaml

def load_config(project_root: Path) -> dict:
    """
    Load the JWST Coma pipeline YAML configuration.
    """
    config_file = project_root / "config" / "pipeline_config.yml"

    if not config_file.exists():
        raise FileNotFoundError(
            f"Pipeline configuration not found: {config_file}"
        )

    with open(config_file, "r") as f:
        return yaml.safe_load(f)

#--------------------------------------------------------------------------------------------#

from pathlib import Path
from typing import Iterable

FITS_SUFFIXES = (
    ".fits",
    ".fit",
    ".fits.gz",
    ".fit.gz",
)

def is_fits_file(path: Path) -> bool:
    """Return True when a path has a recognised FITS filename suffix."""
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in FITS_SUFFIXES)


def discover_fits_files(
    root: str | Path,
    recursive: bool = True,
) -> list[Path]:
    """
    Find FITS files beneath a directory.

    Parameters
    ----------
    root
        Directory to search.
    recursive
        Search subdirectories when True.

    Returns
    -------
    list of pathlib.Path
        Sorted absolute FITS paths.
    """
    root = Path(root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    iterator: Iterable[Path]
    iterator = root.rglob("*") if recursive else root.glob("*")

    return sorted(
        path for path in iterator
        if path.is_file() and is_fits_file(path)
    )

#--------------------------------------------------------------------------------------------#
from typing import Any
from astropy.io import fits

def first_header_value(
    headers: list[fits.Header],
    *keywords: str,
    default: Any = None,
) -> Any:
    """
    Return the first non-empty value found for a sequence of keywords.

    Headers are searched in the order supplied.
    """
    for keyword in keywords:
        for header in headers:
            value = header.get(keyword)

            if value not in (None, ""):
                return value

    return default

#--------------------------------------------------------------------------------------------#

def find_science_hdu(hdul: fits.HDUList) -> int | None:
    """
    Return the index of the most likely science image HDU.
    """
    preferred_names = ("SCI", "PRIMARY")

    for name in preferred_names:
        for index, hdu in enumerate(hdul):
            if hdu.name.upper() == name and hdu.data is not None:
                return index

    for index, hdu in enumerate(hdul):
        if hdu.data is not None and getattr(hdu.data, "ndim", 0) >= 2:
            return index

    return None

#--------------------------------------------------------------------------------------------#

KNOWN_PRODUCT_SUFFIXES = (
    "uncal",
    "rate",
    "rateints",
    "cal",
    "calints",
    "i2d",
    "s2d",
    "x1d",
    "x1dints",
    "crf",
    "crfints",
    "segm",
    "cat",
)


def infer_product_suffix(path: Path) -> str | None:
    """Infer the JWST product suffix from a filename."""
    stem = path.name.lower()

    if stem.endswith(".gz"):
        stem = stem[:-3]

    if stem.endswith(".fits"):
        stem = stem[:-5]
    elif stem.endswith(".fit"):
        stem = stem[:-4]

    final_token = stem.rsplit("_", maxsplit=1)[-1]

    if final_token in KNOWN_PRODUCT_SUFFIXES:
        return final_token

    return None

#--------------------------------------------------------------------------------------------#

def infer_calibration_level(suffix: str | None) -> str | None:
    mapping = {
        "uncal": "Level 1 input",
        "rate": "Level 2a",
        "rateints": "Level 2a",
        "cal": "Level 2b",
        "calints": "Level 2b",
        "i2d": "Level 3",
        "s2d": "Level 2b resampled",
        "x1d": "Level 3 spectrum",
        "x1dints": "Level 2/3 spectrum",
    }
    return mapping.get(suffix)

#--------------------------------------------------------------------------------------------#

def infer_science_filter(
    filter_value,
    pupil_value,
) -> str | None:
    """
    Infer the science filter from JWST FILTER/PUPIL metadata.

    JWST instruments may place the science filter in either FILTER
    or PUPIL. For example:

        NIRCam: FILTER=F150W, PUPIL=CLEAR
        NIRISS: FILTER=CLEAR, PUPIL=F150W

    The raw FILTER and PUPIL values should still be retained separately.
    """
    clear_values = {
        "",
        "CLEAR",
        "CLEARP",
        "NONE",
        "N/A",
    }

    values = []

    for value in (filter_value, pupil_value):
        if value is None:
            continue

        value = str(value).strip().upper()

        if value not in clear_values:
            values.append(value)

    if len(values) == 0:
        return None

    if len(values) == 1:
        return values[0]

    # Preserve unexpected multi-element configurations rather than guessing.
    return ";".join(values)

#--------------------------------------------------------------------------------------------#


def read_fits_metadata(
    path: str | Path,
    project_root: str | Path | None = None,
) -> tuple[dict, list[dict]]:
    """
    Read file-level and HDU-level metadata from one FITS file.

    Returns
    -------
    file_record
        One dictionary describing the FITS product.
    hdu_records
        One dictionary per HDU.
    """
    path = Path(path).expanduser().resolve()
    project_root = (
        Path(project_root).resolve()
        if project_root is not None
        else None
    )

    stat = path.stat()
    suffix = infer_product_suffix(path)

    file_record = {
        "filename": path.name,
        "path": str(path),
        "relative_path": (
            str(path.relative_to(project_root))
            if project_root is not None and path.is_relative_to(project_root)
            else str(path)
        ),
        "size_mb": stat.st_size / 1024**2,
        "product_suffix": suffix,
        "calibration_level": infer_calibration_level(suffix),
        "read_ok": False,
        "error": None,
    }

    hdu_records = []

    try:
        with fits.open(
            path,
            mode="readonly",
            memmap=True,
            lazy_load_hdus=True,
        ) as hdul:

            primary_header = hdul[0].header
            science_index = find_science_hdu(hdul)

            science_header = (
                hdul[science_index].header
                if science_index is not None
                else fits.Header()
            )

            headers = [primary_header, science_header]

            science_data = (
                hdul[science_index].data
                if science_index is not None
                else None
            )

            shape = (
                tuple(science_data.shape)
                if science_data is not None
                else None
            )

            filter_raw = first_header_value(
                headers, "FILTER"
            )
            
            pupil_raw = first_header_value(
                headers, "PUPIL"
            )
            
            science_filter = infer_science_filter(
                filter_raw,
                pupil_raw,
            )

            file_record.update(
                {
                    "read_ok": True,
                    "n_hdus": len(hdul),
                    "science_hdu": science_index,
                    "science_shape": str(shape) if shape else None,
                    "naxis1": science_header.get("NAXIS1"),
                    "naxis2": science_header.get("NAXIS2"),

                    "telescope": first_header_value(
                        headers, "TELESCOP"
                    ),
                    "instrument": first_header_value(
                        headers, "INSTRUME"
                    ),
                    "detector": first_header_value(
                        headers, "DETECTOR"
                    ),
                    "module": first_header_value(
                        headers, "MODULE"
                    ),
                    "channel": first_header_value(
                        headers, "CHANNEL"
                    ),
                    
                    "filter": science_filter,
                    "filter_raw": filter_raw,
                    "pupil_raw": pupil_raw,
                    
                    "exp_type": first_header_value(
                        headers, "EXP_TYPE"
                    ),
                    "subarray": first_header_value(
                        headers, "SUBARRAY"
                    ),
                    "readpatt": first_header_value(
                        headers, "READPATT"
                    ),
                    "ngroups": first_header_value(
                        headers, "NGROUPS"
                    ),
                    "nints": first_header_value(
                        headers, "NINTS"
                    ),

                    "program": first_header_value(
                        headers, "PROGRAM"
                    ),
                    "observation": first_header_value(
                        headers, "OBSERVTN"
                    ),
                    "visit": first_header_value(
                        headers, "VISIT"
                    ),
                    "visit_group": first_header_value(
                        headers, "VISITGRP"
                    ),
                    "sequence_id": first_header_value(
                        headers, "SEQ_ID"
                    ),
                    "activity_id": first_header_value(
                        headers, "ACT_ID"
                    ),
                    "exposure": first_header_value(
                        headers, "EXPOSURE"
                    ),
                    "asn_id": first_header_value(
                        headers, "ASN_ID"
                    ),

                    "date_obs": first_header_value(
                        headers, "DATE-OBS"
                    ),
                    "time_obs": first_header_value(
                        headers, "TIME-OBS"
                    ),
                    "expstart": first_header_value(
                        headers, "EXPSTART"
                    ),
                    "expmid": first_header_value(
                        headers, "EXPMID"
                    ),
                    "expend": first_header_value(
                        headers, "EXPEND"
                    ),
                    "exptime": first_header_value(
                        headers, "EXPTIME", "EFFEXPTM"
                    ),
                    "duration": first_header_value(
                        headers, "DURATION"
                    ),
                    "ncombine": first_header_value(
                        headers, "NCOMBINE"
                    ),

                    "target_name": first_header_value(
                        headers, "TARGNAME"
                    ),
                    "target_ra": first_header_value(
                        headers, "TARG_RA"
                    ),
                    "target_dec": first_header_value(
                        headers, "TARG_DEC"
                    ),
                    "ra_ref": first_header_value(
                        headers, "RA_REF", "CRVAL1"
                    ),
                    "dec_ref": first_header_value(
                        headers, "DEC_REF", "CRVAL2"
                    ),
                    "pa_aper": first_header_value(
                        headers, "PA_APER"
                    ),
                    "roll_ref": first_header_value(
                        headers, "ROLL_REF"
                    ),
                    "v3i_yang": first_header_value(
                        headers, "V3I_YANG"
                    ),
                    "s_region": first_header_value(
                        headers, "S_REGION"
                    ),

                    "bunit": first_header_value(
                        headers, "BUNIT"
                    ),
                    "photmjsr": first_header_value(
                        headers, "PHOTMJSR"
                    ),
                    "photfnu": first_header_value(
                        headers, "PHOTFNU"
                    ),
                    "pixar_sr": first_header_value(
                        headers, "PIXAR_SR"
                    ),
                    "pixar_a2": first_header_value(
                        headers, "PIXAR_A2"
                    ),

                    "datamodel": first_header_value(
                        headers, "DATAMODL"
                    ),
                    "pipeline_version": first_header_value(
                        headers, "CAL_VER"
                    ),
                    "crds_context": first_header_value(
                        headers, "CRDS_CTX"
                    ),
                }
            )

            # Test whether celestial WCS can be constructed.
            try:
                wcs = WCS(science_header)
                file_record["has_celestial_wcs"] = bool(
                    wcs.has_celestial
                )

                if wcs.has_celestial:
                    scales = proj_plane_pixel_scales(wcs.celestial)
                    file_record["pixel_scale_x_arcsec"] = (
                        float(scales[0] * 3600)
                    )
                    file_record["pixel_scale_y_arcsec"] = (
                        float(scales[1] * 3600)
                    )
                else:
                    file_record["pixel_scale_x_arcsec"] = None
                    file_record["pixel_scale_y_arcsec"] = None

            except Exception:
                file_record["has_celestial_wcs"] = False
                file_record["pixel_scale_x_arcsec"] = None
                file_record["pixel_scale_y_arcsec"] = None

            for index, hdu in enumerate(hdul):
                data = hdu.data
                data_shape = (
                    tuple(data.shape)
                    if data is not None
                    else None
                )

                hdu_records.append(
                    {
                        "filename": path.name,
                        "relative_path": file_record["relative_path"],
                        "hdu_index": index,
                        "hdu_name": hdu.name,
                        "hdu_class": type(hdu).__name__,
                        "n_header_cards": len(hdu.header),
                        "data_shape": (
                            str(data_shape)
                            if data_shape is not None
                            else None
                        ),
                        "data_dtype": (
                            str(data.dtype)
                            if data is not None
                            else None
                        ),
                        "bunit": hdu.header.get("BUNIT"),
                    }
                )

    except Exception as exc:
        file_record["error"] = f"{type(exc).__name__}: {exc}"

    return file_record, hdu_records

#--------------------------------------------------------------------------------------------#

def load_dataframe(path):
    path = Path(path)
    suf = path.suffix.lower()
    if suf == '.csv':
        return pd.read_csv(path)
    if suf in ['.parquet', '.pq']:
        return pd.read_parquet(path)
    if suf in ['.pkl', '.pickle']:
        return pd.read_pickle(path)
    if suf == '.feather':
        return pd.read_feather(path)
    raise ValueError(f'Unsupported dataframe format: {path}')

def require_columns(df, mapping, label):
    missing = [v for v in mapping.values() if (v is not None and v not in df.columns)]
    if missing:
        raise KeyError(f'{label}: missing required columns: {missing}')

#--------------------------------------------------------------------------------------------#

