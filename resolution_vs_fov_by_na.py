# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "numpy>=1.24",
#   "scipy>=1.10",
#   "psfmodels>=0.3",
#   "matplotlib>=3.8",
# ]
# ///
# requires-python stops below 3.13 so that uv picks an interpreter with a
# prebuilt psfmodels wheel; a newer interpreter can force a psfmodels source
# build, which on Windows needs MSVC.
"""Plot the light-sheet and ASLM resolution-vs-FOV curves together, one PDF page per illumination NA.

Notes / limitations (printed to stdout by main(), kept out of the figure
itself):
    - Illumination model: an ideal cylindrical-lens beam -- a scalar 1-D
      Debye integral focused along the detection axis and uniform along y,
      with no finite sheet height. The ASLM curve is flat by construction
      under perfect shutter/beam synchronization.
    - Sampling: dz == dxy with ni0 == ni.
    - Illumination beam: each page's curves come from run_light_sheet_sweep()
      and run_aslm_sweep() below, which use the cylindrical beam
      (ILLUMINATION_BEAM, at ILLUMINATION_FILL_FACTOR). No overlay is added to
      a page, so each legend keeps exactly two entries.

Run it with `uv run resolution_vs_fov_by_na.py`; the figure is written to
output/resolution_vs_fov_by_na.pdf beside this file.
"""

from __future__ import annotations

import inspect
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psfmodels as pm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import fftconvolve

OUTPUT_NAME = "resolution_vs_fov_by_na.pdf"

# Axis labels shared by every page.
X_LABEL = "position along beam propagation axis (um)"
Y_LABEL = "system-PSF axial FWHM (um)"

LIGHT_SHEET_LABEL = "static light sheet"
ASLM_LABEL_TEMPLATE = "ASLM (slit width {:.2f} um)"

# Illumination-model disclosure, printed to stdout by main().
ILLUMINATION_MODEL_NOTE = (
    "Illumination model: the light-sheet and ASLM curves both use an ideal "
    "cylindrical-lens beam (scalar 1-D Debye integral), focused along the "
    "detection axis and uniform along y with no finite sheet height. "
    "The ASLM curve is flat by construction under perfect shutter/beam "
    "synchronization. Sampling: dz == dxy with ni0 == ni."
)

# ---------------------------------------------------------------------------
# Sweep constants and the two sweep functions. The constants below are shared
# by both sweeps.
# ---------------------------------------------------------------------------

# Detection NA held fixed across the whole illumination_na sweep and named in
# the figure, shared by run_light_sheet_sweep() and run_aslm_sweep() below so
# the two curves on a page are directly comparable. It is threaded into the
# measurement below -- the reported quantity is the system-PSF (detection x
# effective illumination) axial FWHM, not an illumination-only width. 1.1 is
# the chosen operating point.
DETECTION_NA = 1.1

# Six points -- the 0.1-0.45 band plus 0.6 -- shared by run_light_sheet_sweep()
# and run_aslm_sweep() below so the two sweeps share the same x axis. This is
# the aperture (pupil-edge) NA; the legend keeps naming it.
NA_SWEEP: tuple[float, ...] = (0.10, 0.15, 0.25, 0.35, 0.45, 0.6)

# The cylindrical (one-axis focused, uniform along y) illumination beam used
# by both sweeps. NA_SWEEP stays the aperture NA.
ILLUMINATION_BEAM = "cylindrical"

# The 1-D Gaussian apodization of the cylindrical pupil,
# A(s) = exp(-2 (s / ILLUMINATION_FILL_FACTOR)^2) across the 1-D aperture.
ILLUMINATION_FILL_FACTOR = 0.7

# One copy of these constants is shared by both sweeps.
#
# ni0 equals ni to avoid the psfmodels default-ni0 aberration; dz equals dxy
# because the cardinal rot90 path relabels pre-rotation axes without
# resampling instead of resampling them; psf_size_z=61 is the per-position
# window, because the detection PSF confines the system-PSF axial profile to
# about the depth of focus, so a much larger window is unnecessary here.
COMMON = {
    "wavelength": 0.561,
    "ni": 1.33,
    "ns": 1.33,
    "ni0": 1.33,
    "dxy": 0.108,
    "dz": 0.108,
    "psf_size_z": 61,
    "psf_size_xy": 128,
}

# The FOV positions at which the system-PSF axial FWHM is measured, one real
# generate_psf_seed(emitter_offset=p) call per position (via
# measure_light_sheet_system_fwhm_profile). Both sweeps use the same
# positions so they share one x axis. +-50 um because, measured with the
# pencil beam, the light-sheet curve became erratic beyond about 1-2 Rayleigh
# ranges (pencil-beam Fresnel structure) and at |p| >= 100 um fell below its
# own waist value. 201 points at 0.5 um steps, containing exactly 0.0. The
# step was set against the pencil beam's on-axis illumination minima (about
# 3.9 um apart at NA 0.6) and was not re-derived for the cylindrical beam.
FOV_POSITIONS_UM: tuple[float, ...] = tuple(float(v) for v in np.linspace(-50.0, 50.0, 201))

# The Rayleigh-range annotation is illumination-only and needs a much larger
# window than the DOF-confined system-PSF measurement above to find the
# sqrt(2)x-waist crossing -- dz=0.300 with 1335 slices spans +-200.1 um once
# re-centred. The annotation measures the cylindrical beam's sheet thickness
# (its focused-axis FWHM, illumination_beam=ILLUMINATION_BEAM), so the FOV
# boundary is the root-two growth point of the same beam the curve uses.
RAYLEIGH_WINDOW = {"dz": 0.300, "psf_size_z": 1335}

# Fixed across the whole sweep -- slit_width is not the swept variable.
SLIT_WIDTH = 2.0


def run_light_sheet_sweep(
    nas: tuple[float, ...] = NA_SWEEP,
    positions_um: tuple[float, ...] | None = None,
) -> list[tuple[float, np.ndarray, np.ndarray, tuple[float, float, float] | None]]:
    """Measure the light-sheet system-PSF axial FWHM at each swept NA and annotate its Rayleigh boundary.

    Returns a list of `(na, positions_um, fwhm_um, rayleigh)` quadruples in
    the requested order. The curve itself comes from
    `measure_light_sheet_system_fwhm_profile(positions_um=positions,
    detection_na=DETECTION_NA, illumination_na=na,
    illumination_beam=ILLUMINATION_BEAM,
    illumination_fill_factor=ILLUMINATION_FILL_FACTOR, **COMMON)` -- an
    emitter at FOV position p sits off the illumination waist by p along
    propagation. `positions_um` defaults to the module-level
    `FOV_POSITIONS_UM`, read at call time; the returned positions are already
    centred on the waist home, so no re-centring offset applies to the curve.

    `rayleigh` is separately measured through the illumination-only
    `measure_beam_width_profile(**RAYLEIGH_WINDOW)` (used for annotation
    only) over its own, much larger window -- it is either a
    `(waist_um, left_um, right_um)` triple re-centred by that window's own
    offset (`positions_um[-1] / 2`) onto the same axis as the curve, or None
    when the locator raised `ValueError` for this NA. A rejected NA keeps its
    ordered slot in the returned list and the sweep continues with the
    remaining points.
    """
    if positions_um is None:
        positions_um = FOV_POSITIONS_UM
    positions = np.asarray(positions_um, dtype=np.float64)
    results: list[tuple[float, np.ndarray, np.ndarray, tuple[float, float, float] | None]] = []
    for na in nas:
        centered_um, fwhm_um = measure_light_sheet_system_fwhm_profile(
            positions_um=positions,
            detection_na=DETECTION_NA,
            illumination_na=na,
            illumination_beam=ILLUMINATION_BEAM,
            illumination_fill_factor=ILLUMINATION_FILL_FACTOR,
            **COMMON,
        )

        rayleigh_positions_um, rayleigh_widths_um = measure_beam_width_profile(
            illumination_na=na,
            wavelength=COMMON["wavelength"],
            ni=COMMON["ni"],
            ns=COMMON["ns"],
            dxy=COMMON["dxy"],
            psf_size_xy=COMMON["psf_size_xy"],
            illumination_beam=ILLUMINATION_BEAM,
            illumination_fill_factor=ILLUMINATION_FILL_FACTOR,
            **RAYLEIGH_WINDOW,
        )
        offset = rayleigh_positions_um[-1] / 2.0
        try:
            waist_um, left_um, right_um = locate_rayleigh_range(
                rayleigh_positions_um, rayleigh_widths_um
            )
        except ValueError as exc:
            print(f"skipped Rayleigh annotation for illumination_na={na!r}: {exc}")
            rayleigh = None
        else:
            # Subtract the same offset used to centre the Rayleigh window so
            # the annotation lands on the same centred axis as the curve.
            rayleigh = (waist_um - offset, left_um - offset, right_um - offset)
        results.append((na, centered_um, fwhm_um, rayleigh))
    return results


def run_aslm_sweep(
    nas: tuple[float, ...] = NA_SWEEP,
    positions_um: tuple[float, ...] | None = None,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Measure the ASLM system-PSF axial FWHM at each swept illumination NA.

    Returns a list of `(na, positions_um, fwhm_um)` triples in the requested
    order, each measured through
    `measure_aslm_system_fwhm_profile(positions_um=positions,
    detection_na=DETECTION_NA, illumination_na=na, slit_width=SLIT_WIDTH,
    illumination_beam=ILLUMINATION_BEAM,
    illumination_fill_factor=ILLUMINATION_FILL_FACTOR, **COMMON)`.
    `positions_um` defaults to the module-level `FOV_POSITIONS_UM`, read at
    call time (not captured as a default-argument value). The returned
    positions are already centred on the waist home, so no re-centring offset
    applies here.
    """
    if positions_um is None:
        positions_um = FOV_POSITIONS_UM
    positions = np.asarray(positions_um, dtype=np.float64)
    results: list[tuple[float, np.ndarray, np.ndarray]] = []
    for na in nas:
        centered_um, fwhm_um = measure_aslm_system_fwhm_profile(
            positions_um=positions,
            detection_na=DETECTION_NA,
            illumination_na=na,
            slit_width=SLIT_WIDTH,
            illumination_beam=ILLUMINATION_BEAM,
            illumination_fill_factor=ILLUMINATION_FILL_FACTOR,
            **COMMON,
        )
        results.append((na, centered_um, fwhm_um))
    return results


def pair_sweeps() -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Pair the light-sheet and ASLM sweeps' output by NA, never re-deriving the sweep.

    Returns one `(na, ls_centered, ls_widths, aslm_centered, aslm_widths)`
    entry per value of `NA_SWEEP`, in that order.
    """
    # One shared copy of NA_SWEEP and COMMON feeds both sweeps.

    # Reuse each sweep at its defaults. Indexed positionally because the
    # light-sheet list yields quadruples (na, centered, widths, rayleigh) and
    # the ASLM list yields triples (na, centered, widths).
    light_sheet_results = run_light_sheet_sweep()
    aslm_results = run_aslm_sweep()

    light_sheet_by_na = {result[0]: result for result in light_sheet_results}
    aslm_by_na = {result[0]: result for result in aslm_results}

    missing_from_aslm = set(light_sheet_by_na) - set(aslm_by_na)
    missing_from_light_sheet = set(aslm_by_na) - set(light_sheet_by_na)
    if missing_from_aslm or missing_from_light_sheet:
        raise ValueError(
            "NA values are not paired between the two sweeps: "
            f"missing_from_aslm={sorted(missing_from_aslm)!r} "
            f"missing_from_light_sheet={sorted(missing_from_light_sheet)!r}"
        )

    paired: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for na in NA_SWEEP:
        ls_result = light_sheet_by_na[na]
        aslm_result = aslm_by_na[na]
        paired.append((na, ls_result[1], ls_result[2], aslm_result[1], aslm_result[2]))
    return paired


def build_na_page(
    na: float,
    ls_centered: np.ndarray,
    ls_widths: np.ndarray,
    aslm_centered: np.ndarray,
    aslm_widths: np.ndarray,
    *,
    half_extent: float,
    detection_na: float,
    slit_width: float,
) -> plt.Figure:
    """Build one page pairing the light-sheet and ASLM curves for a single illumination NA.

    Both curves are always plotted, so the legend always carries exactly two
    entries, even when one curve holds no finite point. Widths are plotted
    exactly as measured, NaN entries included: matplotlib renders those as
    honest breaks in the line, and those gaps are never closed.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ls_centered, ls_widths, label=LIGHT_SHEET_LABEL)
    ax.plot(aslm_centered, aslm_widths, label=ASLM_LABEL_TEMPLATE.format(slit_width))

    no_data_notes = []
    if not np.isfinite(ls_widths).any():
        no_data_notes.append(LIGHT_SHEET_LABEL)
    if not np.isfinite(aslm_widths).any():
        no_data_notes.append("ASLM")
    if no_data_notes:
        ax.text(
            0.5,
            0.5,
            f"no measurable width for: {', '.join(no_data_notes)}",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_xlim(-half_extent, half_extent)
    ax.legend()
    # Deliberately left autoscaled per page (not shared across pages): both
    # curves on a page share one y scale, which is what the pairing needs;
    # forcing one scale across all pages would flatten the low-NA pages
    # against the high-NA range.
    ax.set_title(
        f"Resolution vs. FOV at illumination NA {na:.2f}\n"
        f"detection NA {detection_na:.2f}, slit width {slit_width:.2f} um, "
        f"FOV +-{half_extent:.1f} um"
    )
    fig.tight_layout()
    return fig


def write_by_na_pdf(figures: list[plt.Figure], output_path: Path) -> int:
    """Write each figure as one PDF page and return the resulting page count."""
    with PdfPages(output_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
        return pdf.get_pagecount()


def main() -> None:
    """Run both sweeps, pair them, and write the combined by-NA PDF."""
    # The illumination-model disclosure is printed to stdout (and kept in the
    # module docstring's Notes / limitations section) rather than rendered on
    # the figures, so the figures stay clean.
    print(ILLUMINATION_MODEL_NOTE)

    paired = pair_sweeps()
    half_extent = max(abs(v) for v in FOV_POSITIONS_UM)

    for na, ls_centered, ls_widths, aslm_centered, aslm_widths in paired:
        ls_finite = int(np.isfinite(ls_widths).sum())
        aslm_finite = int(np.isfinite(aslm_widths).sum())
        print(
            f"illumination_na={na:.2f}: light_sheet finite={ls_finite}/{ls_widths.size}, "
            f"aslm finite={aslm_finite}/{aslm_widths.size}"
        )

    figures = [
        build_na_page(
            na,
            ls_centered,
            ls_widths,
            aslm_centered,
            aslm_widths,
            half_extent=half_extent,
            detection_na=DETECTION_NA,
            slit_width=SLIT_WIDTH,
        )
        for na, ls_centered, ls_widths, aslm_centered, aslm_widths in paired
    ]

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME
    page_count = write_by_na_pdf(figures, output_path)
    print(f"wrote {output_path} ({page_count} pages)")


# ---------------------------------------------------------------------------
# System-PSF and beam-width measurement: axial FWHM profiles and Rayleigh-range
# location.
# ---------------------------------------------------------------------------

# Relative tolerance (1e-6) for float32 noise when comparing adjacent samples
# in _bound_to_first_local_minimum.
_LOBE_MIN_RTOL = 1e-6

# The two failure-reason tags, grouped in one warning per call. "No local
# minimum" is the more fundamental failure.
_NO_LOCAL_MINIMUM = "no local minimum before array edge"
_LOBE_TOO_NARROW = "central lobe too narrow to reach half-max"


def _bound_to_first_local_minimum(
    profile: np.ndarray, indices: np.ndarray, peak_value: float
) -> tuple[np.ndarray, bool]:
    """Truncate an outward walk at the first flanking local minimum of the lobe.

    A plateau is entered by a strict descent and confirmed as the
    lobe-bounding minimum only once an above-threshold rise follows it --
    the run of adjacent samples in between, however long, all tie within
    `_LOBE_MIN_RTOL * peak_value` of one another. Truncation is inclusive
    of the LAST plateau sample: the one immediately before the
    confirming rise, not just the first descended-into sample. A
    single-sample run (no plateau extension) is the simple strict-descent
    case.
    """
    # The threshold is computed once per call from the CURRENT module global
    # (read at call time, never captured as a default-argument value) -- the
    # fixed per-slice reference, not a per-step recomputation relative to a
    # drifting local value.
    threshold = _LOBE_MIN_RTOL * peak_value
    for i in range(1, indices.size - 1):
        if profile[indices[i - 1]] > profile[indices[i]]:
            # Strict descent into i opens a candidate plateau run starting
            # at j = i. Extend j while the next step stays within
            # threshold of a tie; a one-sample run (j stays at i) is the
            # simple strict-descent case.
            j = i
            while (
                j + 1 < indices.size
                and abs(profile[indices[j + 1]] - profile[indices[j]]) <= threshold
            ):
                j += 1
            if (
                j + 1 < indices.size
                and profile[indices[j + 1]] - profile[indices[j]] > threshold
            ):
                # Truncation is inclusive of the last plateau sample:
                # when the minimum itself is the first below-half-max
                # sample, excluding it would drop a valid crossing.
                return indices[: j + 1], True
            # No above-threshold rise confirmed this run (array edge
            # reached while still flat, or the very next step is itself
            # another descent) -- not a minimum yet, keep scanning from
            # i + 1.
    # The last index can never be a minimum -- no following sample confirms
    # a rise -- so a walk that reaches the array edge without finding one
    # falls back to the full, unmodified array. There is deliberately no
    # global scan for a deeper minimum further out.
    return indices, False


def _half_max_width_with_reason(
    profile: np.ndarray, peak_index: int, spacing: float
) -> tuple[float | None, str | None]:
    """Return `(width, reason)`: the central-lobe-bounded half-max width, or `(None, reason)`.

    Bounds each side's outward walk to the first flanking local minimum
    before running the first-below-half-max crossing search with linear
    interpolation. Returns `(None, reason)` when the peak is <= 0 or either
    bounded side never drops below half max, classified by priority: a
    reason of "no local minimum before array edge" if a failed side also
    never found a bounding local minimum, otherwise "central lobe too
    narrow to reach half-max".
    """
    peak_value = profile[peak_index]
    if peak_value <= 0:
        return None, _NO_LOCAL_MINIMUM
    half_max = peak_value / 2.0

    def _crossing(indices: np.ndarray) -> float | None:
        values = profile[indices]
        below = np.where(values < half_max)[0]
        if below.size == 0:
            return None
        edge = below[0]
        i0, i1 = indices[edge - 1], indices[edge]
        v0, v1 = profile[i0], profile[i1]
        frac = (half_max - v0) / (v1 - v0)
        return i0 + frac * (i1 - i0)

    left_indices = np.arange(peak_index, -1, -1)
    right_indices = np.arange(peak_index, profile.size)

    left_bounded, left_found_minimum = _bound_to_first_local_minimum(
        profile, left_indices, peak_value
    )
    right_bounded, right_found_minimum = _bound_to_first_local_minimum(
        profile, right_indices, peak_value
    )
    left_crossing = _crossing(left_bounded)
    right_crossing = _crossing(right_bounded)
    if left_crossing is None or right_crossing is None:
        # Priority to the more fundamental failure.
        no_minimum_on_a_failed_side = (
            left_crossing is None and not left_found_minimum
        ) or (right_crossing is None and not right_found_minimum)
        reason = _NO_LOCAL_MINIMUM if no_minimum_on_a_failed_side else _LOBE_TOO_NARROW
        return None, reason
    return (right_crossing - left_crossing) * spacing, None


def _axial_fwhm_with_reason(seed: np.ndarray, dz: float) -> tuple[float | None, str | None]:
    """Return `(width, reason)` for the axial (Z) FWHM of a (Z, Y, X) PSF seed."""
    peak_z, peak_y, peak_x = np.unravel_index(np.argmax(seed), seed.shape)
    profile = seed[:, peak_y, peak_x].astype(np.float64)
    return _half_max_width_with_reason(profile, peak_z, dz)


def _validate_common(caller: str, values: dict) -> None:
    """Raise ValueError naming every missing/non-positive parameter, before any generation."""
    missing = [name for name, value in values.items() if value is None or value <= 0]
    if missing:
        raise ValueError(
            f"Missing or non-positive parameter(s) for {caller}: " + ", ".join(missing)
        )


def _require_cubic_voxels(caller: str, dxy: float, dz: float) -> None:
    """Raise ValueError when the requested voxel grid is not cubic (dz == dxy)."""
    # The cardinal rotation (_legacy_rot90_rotation) relabels pre-rotation axes
    # with a bare np.rot90 and no resampling, which is exact only when
    # dz == dxy.
    if not np.isclose(dz, dxy, rtol=1e-9, atol=0.0):
        raise ValueError(
            f"{caller}: dz ({dz!r}) must equal dxy ({dxy!r}) -- the cardinal "
            "rotation relabels pre-rotation axes without resampling, so anisotropic "
            "voxels mis-scale the rotated beam"
        )


def _validate_positions(caller: str, positions_um) -> np.ndarray:
    """Return `positions_um` as a validated, non-empty, 1-D, finite float64 array."""
    positions = np.asarray(positions_um, dtype=np.float64)
    if positions.ndim != 1 or positions.size == 0 or not np.all(np.isfinite(positions)):
        raise ValueError(
            f"{caller}: positions_um must be a non-empty 1-D sequence of finite "
            f"values, got {positions_um!r}"
        )
    return positions


def _measure_profile(
    caller: str, psf_mode: str, positions: np.ndarray, seed_kwargs: dict
) -> np.ndarray:
    """Measure one system-PSF axial FWHM per FOV position in `positions`.

    Each position is a real `generate_psf_seed` call with `emitter_offset=p`
    -- no short-circuiting to a single call, so ASLM's flatness is
    observed, not assumed. A None result stays NaN; never clamped or
    extrapolated. Unmeasurable positions are grouped by reason into one
    warning: "no local minimum before array edge" and "central
    lobe too narrow to reach half-max", each listing its own positions,
    empty groups omitted.
    """
    dz = seed_kwargs["dz"]
    fwhm_um = np.full(positions.shape, np.nan, dtype=np.float64)
    failed_no_local_minimum: list[float] = []
    failed_lobe_too_narrow: list[float] = []
    for i, p in enumerate(positions):
        seed = generate_psf_seed(psf_mode=psf_mode, emitter_offset=float(p), **seed_kwargs)
        value, reason = _axial_fwhm_with_reason(seed, dz)
        if value is None:
            # Round for display only -- positions_um itself stays exact.
            position_um = round(float(p), 6)
            if reason == _NO_LOCAL_MINIMUM:
                failed_no_local_minimum.append(position_um)
            else:
                failed_lobe_too_narrow.append(position_um)
            continue
        fwhm_um[i] = value
    message_parts = []
    if failed_no_local_minimum:
        message_parts.append(f"{_NO_LOCAL_MINIMUM} at (um): {failed_no_local_minimum!r}")
    if failed_lobe_too_narrow:
        message_parts.append(f"{_LOBE_TOO_NARROW} at (um): {failed_lobe_too_narrow!r}")
    if message_parts:
        warnings.warn(f"{caller}: " + "; ".join(message_parts), stacklevel=3)
    return fwhm_um


def measure_light_sheet_system_fwhm_profile(
    *,
    positions_um,
    detection_na: float,
    illumination_na: float,
    wavelength: float,
    ni: float,
    ns: float,
    ni0: float | None,
    dxy: float,
    dz: float,
    psf_size_z: int = 61,
    psf_size_xy: int = 128,
    polar_deg: float = 90.0,
    azimuthal_deg: float = 0.0,
    tg: float | None = None,
    tg0: float | None = None,
    ng: float | None = None,
    ng0: float | None = None,
    ti0: float | None = None,
    oversample_factor: int = 3,
    psf_model: str = "vectorial",
    illumination_fill_factor: float | None = None,
    illumination_beam: str = "pencil",
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the light-sheet system-PSF axial FWHM at each FOV position.

    An emitter at FOV position `p` sits off the illumination waist by `p`
    along the beam's propagation direction -- each position is
    composed through `generate_psf_seed(psf_mode="light_sheet",
    emitter_offset=p, ...)`, never through psfmodels' own SPIM
    total-system-PSF function or a local re-composition. The reported
    quantity is the system-PSF (detection
    x rotated illumination) axial FWHM through the seed's global 3-D peak.

    `ni0` has no default: every caller must choose explicitly whether to
    pass a value or `None`.

    `illumination_fill_factor` defaults to `None` (hard-aperture
    pupil); when set, the illumination arm is the
    Gaussian-apodized pupil, forwarded to every per-position
    `generate_psf_seed` call.

    `illumination_beam` defaults to `"pencil"`; with
    `"cylindrical"` it is forwarded to every per-position seed, so the
    illumination arm is the cylindrical beam. An unknown value is rejected by
    `generate_psf_seed` before any PSF is generated.

    Returns `(positions_um, fwhm_um)`, both float64 and of equal length.
    Unmeasurable positions are NaN with a single named `UserWarning`.
    """
    _validate_common(
        "measure_light_sheet_system_fwhm_profile",
        {
            "detection_na": detection_na,
            "illumination_na": illumination_na,
            "wavelength": wavelength,
            "ni": ni,
            "ns": ns,
            "dxy": dxy,
            "dz": dz,
            "psf_size_z": psf_size_z,
            "psf_size_xy": psf_size_xy,
        },
    )
    _require_cubic_voxels("measure_light_sheet_system_fwhm_profile", dxy, dz)
    positions = _validate_positions("measure_light_sheet_system_fwhm_profile", positions_um)
    seed_kwargs = dict(
        na=detection_na,
        detection_na=detection_na,
        illumination_na=illumination_na,
        wavelength=wavelength,
        ni=ni,
        ns=ns,
        ni0=ni0,
        tg=tg,
        tg0=tg0,
        ng=ng,
        ng0=ng0,
        ti0=ti0,
        oversample_factor=oversample_factor,
        psf_model=psf_model,
        dxy=dxy,
        dz=dz,
        psf_size_z=psf_size_z,
        psf_size_xy=psf_size_xy,
        background=0.0,
        polar_deg=polar_deg,
        azimuthal_deg=azimuthal_deg,
        illumination_fill_factor=illumination_fill_factor,
        illumination_beam=illumination_beam,
    )
    fwhm_um = _measure_profile(
        "measure_light_sheet_system_fwhm_profile", "light_sheet", positions, seed_kwargs
    )
    return positions.copy(), fwhm_um


def measure_aslm_system_fwhm_profile(
    *,
    positions_um,
    detection_na: float,
    illumination_na: float,
    wavelength: float,
    ni: float,
    ns: float,
    ni0: float | None,
    dxy: float,
    dz: float,
    slit_width: float,
    psf_size_z: int = 61,
    psf_size_xy: int = 128,
    polar_deg: float = 90.0,
    azimuthal_deg: float = 0.0,
    tg: float | None = None,
    tg0: float | None = None,
    ng: float | None = None,
    ng0: float | None = None,
    ti0: float | None = None,
    oversample_factor: int = 3,
    psf_model: str = "vectorial",
    illumination_fill_factor: float | None = None,
    illumination_beam: str = "pencil",
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the ASLM system-PSF axial FWHM at each FOV position.

    The rolling shutter is assumed perfectly synchronized to the swept
    beam waist, so the ASLM waist sweeps the whole simulated window -- there
    is no sweep-range parameter, and the curve is expected to be flat except
    for window-edge truncation. Each position is a real
    `generate_psf_seed(psf_mode="aslm", emitter_offset=p, slit_width=...)`
    call; the flatness is observed here, not assumed or short-circuited to a
    single call.

    `illumination_fill_factor` defaults to `None` (hard-aperture
    pupil); when set, the illumination arm is the
    Gaussian-apodized pupil, forwarded to every per-position
    `generate_psf_seed` call.

    `illumination_beam` defaults to `"pencil"`; with
    `"cylindrical"` it is forwarded to every per-position seed, so the
    illumination arm is the cylindrical beam. An unknown value is rejected by
    `generate_psf_seed` before any PSF is generated.

    Returns `(positions_um, fwhm_um)`, both float64 and of equal length.
    """
    _validate_common(
        "measure_aslm_system_fwhm_profile",
        {
            "detection_na": detection_na,
            "illumination_na": illumination_na,
            "wavelength": wavelength,
            "ni": ni,
            "ns": ns,
            "dxy": dxy,
            "dz": dz,
            "psf_size_z": psf_size_z,
            "psf_size_xy": psf_size_xy,
        },
    )
    _require_cubic_voxels("measure_aslm_system_fwhm_profile", dxy, dz)
    if slit_width is None or slit_width <= 0:
        raise ValueError(
            f"measure_aslm_system_fwhm_profile: slit_width must be > 0, got {slit_width!r}"
        )
    positions = _validate_positions("measure_aslm_system_fwhm_profile", positions_um)
    seed_kwargs = dict(
        na=detection_na,
        detection_na=detection_na,
        illumination_na=illumination_na,
        wavelength=wavelength,
        ni=ni,
        ns=ns,
        ni0=ni0,
        tg=tg,
        tg0=tg0,
        ng=ng,
        ng0=ng0,
        ti0=ti0,
        oversample_factor=oversample_factor,
        psf_model=psf_model,
        dxy=dxy,
        dz=dz,
        psf_size_z=psf_size_z,
        psf_size_xy=psf_size_xy,
        background=0.0,
        polar_deg=polar_deg,
        azimuthal_deg=azimuthal_deg,
        slit_width=slit_width,
        illumination_fill_factor=illumination_fill_factor,
        illumination_beam=illumination_beam,
    )
    fwhm_um = _measure_profile(
        "measure_aslm_system_fwhm_profile", "aslm", positions, seed_kwargs
    )
    return positions.copy(), fwhm_um


def measure_beam_width_profile(
    *,
    illumination_na: float,
    wavelength: float,
    ni: float,
    ns: float,
    dxy: float,
    dz: float,
    psf_size_z: int = 61,
    psf_size_xy: int = 128,
    illumination_fill_factor: float | None = None,
    illumination_beam: str = "pencil",
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the illumination beam's transverse (Y) FWHM at every Z position.

    Returns `(positions_um, widths_um)`: two parallel float64 arrays, each of
    length `psf_size_z`. `positions_um[i] == i * dz`, in micrometres,
    strictly ascending. `widths_um[i]` is the interpolated half-max Y-profile
    width measured at `positions_um[i]`, or NaN when no clean half-max
    crossing exists at that position.

    `illumination_fill_factor` defaults to `None` (hard-aperture
    pupil); when set, the measured beam is the Gaussian-apodized
    illumination pupil, so the Rayleigh-range annotation uses the same beam
    as its curve.

    With `illumination_beam="cylindrical"` the measured beam is the
    cylindrical (one-axis focused, uniform along y) illumination arm and each
    width is its focused-axis FWHM, the sheet thickness, so the Rayleigh
    annotation uses the same beam as its curve.
    """
    # Validate this function's OWN parameters before spending time on PSF
    # generation. illumination_na is validated HERE and nowhere else --
    # generate_theoretical_psf deletes its own illumination_na argument on
    # entry and never validates it, so this is the only guard that will ever
    # catch it. psf_size_z/psf_size_xy are validated here because the
    # downstream failure for a non-positive size is an opaque
    # `ValueError: zero-size array to reduction operation maximum which has
    # no identity` raised from inside NumPy, naming neither the parameter
    # nor the caller.
    required_values = {
        "illumination_na": illumination_na,
        "wavelength": wavelength,
        "ni": ni,
        "ns": ns,
        "dxy": dxy,
        "dz": dz,
        "psf_size_z": psf_size_z,
        "psf_size_xy": psf_size_xy,
    }
    missing = [name for name, value in required_values.items() if value is None or value <= 0]
    if missing:
        raise ValueError(
            "Missing or non-positive parameter(s) for measure_beam_width_profile: "
            + ", ".join(missing)
        )
    if illumination_beam not in ("pencil", "cylindrical"):
        raise ValueError(
            "measure_beam_width_profile: illumination_beam must be 'pencil' or "
            f"'cylindrical', got {illumination_beam!r}"
        )

    # Generate the illumination-arm PSF internally by plugging
    # illumination_na into detection_na -- the same trick generate_psf_seed
    # uses for its own illumination branch. Deliberately never
    # generate_psf_seed(): that returns detection times rotated illumination,
    # which would mix the detection PSF's own axial extent into this
    # measurement and destroy the NA-dependence this measurement is meant to
    # show.
    if illumination_beam == "cylindrical":
        psf = generate_cylindrical_illumination_psf(
            illumination_na=illumination_na,
            wavelength=wavelength,
            ni=ni,
            dxy=dxy,
            dz=dz,
            psf_size_z=psf_size_z,
            psf_size_xy=psf_size_xy,
            illumination_fill_factor=illumination_fill_factor,
        )
        # The cylindrical beam is identical along pre-rotation Y, so the Y
        # profile the loop below reads is flat; swapping hands the loop the
        # focused axis (pre-rotation X), whose FWHM is the sheet thickness.
        psf = np.swapaxes(psf, 1, 2)
    elif illumination_fill_factor is None:
        psf = generate_theoretical_psf(
            detection_na=illumination_na,
            illumination_na=illumination_na,
            wavelength=wavelength,
            ni=ni,
            ns=ns,
            dxy=dxy,
            dz=dz,
            psf_size_z=psf_size_z,
            psf_size_xy=psf_size_xy,
        )
    else:
        raise ValueError(
            "illumination_fill_factor with illumination_beam='pencil' (the Gaussian-apodized "
            "pencil beam) is not supported in this script"
        )
    # psf shape is (psf_size_z, psf_size_xy, psf_size_xy) in (Z, Y, X) order.

    # Fixed global peak, not re-located per Z slice -- psfmodels' illumination
    # PSF has zero lateral (Y/X) peak drift across Z slices near focus, but
    # the per-slice argmax DOES drift on far-from-focus slices whose on-axis
    # intensity has collapsed and whose slice maximum is a diffraction
    # sidelobe (measured at illumination_na=0.6, psf_size_z=21: per-slice
    # (y, x) argmax took four distinct values across the 21 slices). Fixing
    # the peak globally avoids tracking sidelobes instead of beam centre.
    _peak_z, peak_y, peak_x = np.unravel_index(np.argmax(psf), psf.shape)

    # Positions in physical micrometres, voxel_index * dz, so callers need no
    # conversion.
    positions_um = np.arange(psf_size_z, dtype=np.float64) * dz
    widths_um = np.full(psf_size_z, np.nan, dtype=np.float64)
    failed_no_local_minimum: list[float] = []
    failed_lobe_too_narrow: list[float] = []

    # Dense, one measurement per Z voxel, no stride or step parameter.
    #
    # Far-from-focus slices legitimately produce large, non-monotonic
    # widths because the Y profile there carries diffraction sidelobe
    # structure rather than a clean single lobe (measured at
    # illumination_na=0.4: the widths span 0.73 um to 8.48 um across the
    # 61-slice window). The search is bounded to the central lobe, so a
    # ring-merged reading becomes an honest NaN gap tagged "central lobe too
    # narrow" instead of silently including ring structure in the measured
    # width -- no smoothing, clamping, or extrapolation is applied.
    for z in range(psf_size_z):
        # Z is the propagation axis the position sweeps along, and the
        # reported width is the transverse FWHM measured along Y at fixed
        # peak_x. Y and X are equivalent for this rotationally symmetric
        # on-axis PSF model; Y is the chosen convention.
        profile = psf[z, :, peak_x].astype(np.float64)
        peak_value = profile[peak_y]
        if peak_value <= 0:
            # No lobe to bound at all -- more fundamental than a
            # bounded-but-too-narrow lobe, so this goes to the
            # no-local-minimum group. Round for display only --
            # positions_um itself stays exact; float64
            # multiplication of index * dz can produce artifacts like
            # 3 * 0.3 == 0.8999999999999999, which would silently fail to
            # name "0.9" in the warning text below.
            failed_no_local_minimum.append(round(float(positions_um[z]), 6))
            continue
        half_max = peak_value / 2.0

        def _crossing(indices: np.ndarray) -> float | None:
            values = profile[indices]
            below = np.where(values < half_max)[0]
            if below.size == 0:
                return None
            edge = below[0]
            i0, i1 = indices[edge - 1], indices[edge]
            v0, v1 = profile[i0], profile[i1]
            frac = (half_max - v0) / (v1 - v0)
            return i0 + frac * (i1 - i0)

        left_indices = np.arange(peak_y, -1, -1)
        right_indices = np.arange(peak_y, profile.size)
        # Bound each side's outward walk to the first flanking local minimum
        # (rtol-tolerant) before crossing search; the bounded index array is
        # always a prefix of the unbounded one.
        left_bounded, left_found_minimum = _bound_to_first_local_minimum(
            profile, left_indices, peak_value
        )
        right_bounded, right_found_minimum = _bound_to_first_local_minimum(
            profile, right_indices, peak_value
        )
        left_crossing = _crossing(left_bounded)
        right_crossing = _crossing(right_bounded)
        if left_crossing is None or right_crossing is None:
            # Priority to the more fundamental failure. A side that
            # failed to cross AND never found a bounding local minimum
            # means there was no lobe boundary to search within at all --
            # tag the whole position "no local minimum before array edge".
            # Otherwise both sides that failed to cross did find a
            # bounding minimum, so the lobe itself is genuinely too narrow
            # to reach half-max -- "central lobe too narrow to reach
            # half-max".
            no_minimum_on_a_failed_side = (
                left_crossing is None and not left_found_minimum
            ) or (right_crossing is None and not right_found_minimum)
            # Round for display only -- positions_um itself stays exact;
            # float64 multiplication of index * dz can produce
            # artifacts like 3 * 0.3 == 0.8999999999999999, which would
            # silently fail to name "0.9" in the warning text below.
            position_um = round(float(positions_um[z]), 6)
            if no_minimum_on_a_failed_side:
                failed_no_local_minimum.append(position_um)
            else:
                failed_lobe_too_narrow.append(position_um)
            continue
        widths_um[z] = (right_crossing - left_crossing) * dxy

    # Name every failed position, not just the first -- measured at
    # illumination_na=0.6, psf_size_z=21, psf_size_xy=16, index 0 measures
    # cleanly while indices 1 through 5 do not, so the NaN block is neither
    # contiguous with an array edge nor inferable from the array's shape. A
    # caller who only saw "5 positions failed" or "first failure at 0.3 um"
    # could not reconstruct which entries to skip. The array itself stays
    # untouched -- no clamping to the lateral extent, no extrapolation from
    # neighbours, no zero fill, and no dropping of failed entries (which
    # would break the parallel-array length contract); this warning is
    # purely additive surfacing, grouped by reason.
    message_parts = []
    if failed_no_local_minimum:
        message_parts.append(
            f"no local minimum before array edge at (um): {failed_no_local_minimum!r}"
        )
    if failed_lobe_too_narrow:
        message_parts.append(
            "central lobe too narrow to reach half-max at (um): "
            f"{failed_lobe_too_narrow!r}"
        )
    if message_parts:
        warnings.warn(
            "measure_beam_width_profile: " + "; ".join(message_parts),
            stacklevel=2,
        )

    return positions_um, widths_um


def locate_rayleigh_range(
    positions_um: np.ndarray, widths_um: np.ndarray
) -> tuple[float, float, float]:
    """Locate the beam waist and the two sqrt(2)x-waist Rayleigh boundaries.

    Returns `(waist_position_um, left_position_um, right_position_um)`, all
    in micrometres. `left` is toward decreasing Z, `right` toward increasing
    Z.

    Raises `ValueError` when either outward walk reaches an array edge
    without finding the sqrt(2)x-waist crossing.
    """
    # Three positions only, no waist width: the waist width is one
    # np.nanmin(widths_um) call away for any caller that already holds the
    # profile, and appending a width to a tuple of positions makes the
    # unpack order a foot-gun. A plain tuple is used rather than a
    # NamedTuple or dataclass.

    positions_um = np.asarray(positions_um, dtype=np.float64)
    widths_um = np.asarray(widths_um, dtype=np.float64)

    # Validate both array parameters before any waist search runs.
    # Three sequential stages, not one flat list, because later checks are
    # undefined on inputs the earlier ones reject -- you cannot meaningfully
    # diff a 2-D array, and you cannot pair-index arrays of different
    # lengths. Within each stage every offender is collected and named in
    # one message.

    # Stage A -- shape.
    problems: list[str] = []
    if positions_um.ndim != 1:
        problems.append(f"positions_um must be 1-D, got ndim={positions_um.ndim}")
    if widths_um.ndim != 1:
        problems.append(f"widths_um must be 1-D, got ndim={widths_um.ndim}")
    if problems:
        raise ValueError("locate_rayleigh_range: " + "; ".join(problems))

    # Stage B -- pairing.
    problems = []
    if positions_um.shape != widths_um.shape:
        problems.append(
            "positions_um and widths_um must have the same length, got "
            f"positions_um.size={positions_um.size}, widths_um.size={widths_um.size}"
        )
    if positions_um.size == 0:
        problems.append("positions_um and widths_um must not be empty")
    if problems:
        raise ValueError("locate_rayleigh_range: " + "; ".join(problems))

    # Stage C -- content.
    problems = []
    if not np.isfinite(positions_um).all():
        problems.append("positions_um must contain only finite values")
    # This is the contract measure_beam_width_profile already guarantees
    # (positions_um[i] == i * dz), asserted here because the outward walks
    # read positions by index and a scrambled or duplicated position array
    # would produce a boundary that is arithmetically valid and physically
    # meaningless. A length-1 array trivially satisfies this: np.diff of it
    # is empty and np.all([]) is True, which is intended -- a single-sample
    # profile is well-formed input.
    elif not np.all(np.diff(positions_um) > 0):
        problems.append("positions_um must be strictly ascending")

    measured = ~np.isnan(widths_um)
    if not measured.any():
        # Without this guard, the np.nanargmin call below raises NumPy's own
        # `ValueError: All-NaN slice encountered`, which names neither the
        # parameter nor the caller.
        #
        # This message is deliberately kept SEPARATE from the array-too-short
        # message below, because "increase psf_size_z" is actively wrong
        # advice for a profile in which nothing was measurable at any
        # position -- a longer axial array yields more NaN, not a crossing.
        # The measurement's own warning already named every unmeasurable
        # position, so the caller has the detail; what this message adds is
        # the correct remedy: the lateral measurement window, not the axial
        # array.
        problems.append(
            "widths_um is entirely NaN -- every position is unmeasurable, so no "
            "waist can be located; increase psf_size_xy or illumination_na when "
            "generating the beam-width profile"
        )
    else:
        bad = measured & (~np.isfinite(widths_um) | (widths_um <= 0.0))
        if bad.any():
            # The sqrt(2) threshold is computed from the waist width, so a
            # zero waist makes the threshold zero and the first sample
            # examined on each side satisfies the crossing test immediately,
            # returning a boundary equal to the waist position -- a
            # plausible-looking number with no physical meaning.
            offending = [
                round(float(v), 6) for v in positions_um[bad]
            ]
            problems.append(
                "widths_um must be positive and finite at every measured "
                f"position, offending position(s) (um): {offending!r}"
            )

    # NaN among otherwise valid widths is NOT rejected -- it is a
    # documented, legitimate output for an unmeasurable position, and the
    # walk skips it.
    if problems:
        raise ValueError("locate_rayleigh_range: " + "; ".join(problems))

    # The waist is the NaN-ignoring minimum of the measured profile,
    # chosen over any fitted minimum so the number stays grounded in what
    # was actually simulated.
    waist_index = int(np.nanargmin(widths_um))
    waist_width_um = float(widths_um[waist_index])
    waist_position_um = float(positions_um[waist_index])

    # The Rayleigh definition used here -- the FOV boundary is where
    # the beam width has grown sqrt(2)x from its waist.
    threshold_um = float(np.sqrt(2.0)) * waist_width_um

    # First-crossing-wins search, not a global search for the
    # cleanest crossing -- no smoothing, no fitting, and no second pass
    # looking further out.
    def _walk_outward(order: range) -> float | None:
        previous: int | None = None
        for index in order:
            if np.isnan(widths_um[index]):
                # A data gap does not stop the outward search, and a
                # skipped sample is never used as an interpolation
                # endpoint.
                continue
            if widths_um[index] >= threshold_um:
                if previous is None:
                    return None
                v0, v1 = widths_um[previous], widths_um[index]
                p0, p1 = positions_um[previous], positions_um[index]
                # Denominator cannot be zero: v1 >= threshold_um > v0 by
                # construction.
                frac = (threshold_um - v0) / (v1 - v0)
                # Sub-voxel interpolation -- deliberately NOT snapped to a
                # sampled position.
                return float(p0 + frac * (p1 - p0))
            previous = index
        # Reaching the array edge with no crossing -- whether because the
        # widths never grew enough or because the remaining samples were all
        # NaN -- is the array-too-small signal, turned into a ValueError
        # below.
        return None

    # Both ranges start AT the waist index, which is guaranteed non-NaN and
    # guaranteed below the threshold, so `previous` is always populated on
    # the first iteration of a non-empty walk.
    left_position_um = _walk_outward(range(waist_index, -1, -1))
    right_position_um = _walk_outward(range(waist_index, positions_um.size))

    unreachable: list[str] = []
    if left_position_um is None:
        # Round for display only -- positions_um itself stays exact:
        # float64 multiplication of index * dz can produce artifacts like
        # 3 * 0.3 == 0.8999999999999999, and without rounding this message
        # would name that artifact instead of "0.9" or "1.8".
        unreachable.append(
            f"Z={round(float(positions_um[0]), 6)} um (walking toward decreasing Z)"
        )
    if right_position_um is None:
        unreachable.append(
            f"Z={round(float(positions_um[-1]), 6)} um (walking toward increasing Z)"
        )

    if unreachable:
        # This single branch serves both cases -- running out of samples and
        # a NaN run reaching the edge are indistinguishable for this purpose,
        # and `_walk_outward` deliberately returns the same None for both.
        #
        # Deliberately NOT done here -- no clamping to
        # positions_um[0]/positions_um[-1], no linear extrapolation past the
        # sampled range, no mirroring of the side that did succeed, and no
        # NaN return. A caller who receives a number from this function has
        # a number the simulation actually contains. At illumination_na=0.30
        # with a 61-slice window the true crossing lies outside the array on
        # both sides, and clamping would report an 18 um FOV extent that is
        # an artifact of the window size rather than of the optics.
        raise ValueError(
            "locate_rayleigh_range: Rayleigh-range crossing not found before "
            "array edge at " + ", ".join(unreachable) +
            "; increase psf_size_z (or dz) when generating the beam-width "
            "profile."
        )

    # The two sides are reported separately and are never averaged or
    # collapsed into one number here.
    return waist_position_um, left_position_um, right_position_um


# ---------------------------------------------------------------------------
# PSF seed generation: psfmodels-based detection and illumination PSFs,
# cardinal rotation, and the ASLM slit gate.
# ---------------------------------------------------------------------------

RIGHT_ANGLE_TOLERANCE = 1e-6

# Floor on the theta-quadrature interval count -- keeps the composite Simpson
# rule well-resolved even for a wide (near-uniform) pupil where the other three
# N candidates below could all be small.
_APODIZED_MIN_THETA_INTERVALS = 60
# Matches psfmodels' own vectorial sampling density (4 samples per half-cycle
# of the fastest-oscillating term), so the apodized integral resolves phase
# and pupil-weight variation at the same density psfmodels relies on.
_APODIZED_THETA_SAMPLES_PER_HALF_CYCLE = 4
# Resolution guard: caps quadrature cost for a very small fill
# factor or a very large z extent, rather than letting N grow unbounded.
_APODIZED_MAX_THETA_INTERVALS = 8192
# Accepted values of generate_psf_seed's illumination_beam: "pencil" is the
# round-pupil beam (the default), "cylindrical" the 1-D one.
_ILLUMINATION_BEAMS: tuple[str, ...] = ("pencil", "cylindrical")


def normalise_psf(psf: np.ndarray) -> np.ndarray:
    psf = np.nan_to_num(psf.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    psf = np.clip(psf, 0, None)
    total = float(psf.sum())
    if total > 0:
        psf = psf / total
    return psf.astype(np.float32, copy=False)


def generate_theoretical_psf(
    na: float | None = None,
    detection_na: float | None = None,
    illumination_na: float | None = None,
    wavelength: float | None = None,
    ni: float | None = None,
    ns: float | None = None,
    ni0: float | None = None,
    tg: float | None = None,
    tg0: float | None = None,
    ng: float | None = None,
    ng0: float | None = None,
    ti0: float | None = None,
    oversample_factor: int = 3,
    psf_model: str = "vectorial",
    dxy: float | None = None,
    dz: float | None = None,
    psf_size_z: int = 61,
    psf_size_xy: int = 128,
    background: float = 0.0,
    focus_offset: float = 0.0,
) -> np.ndarray:
    """Generate a normalized 3-D PSF seed with ``psfmodels.make_psf``.

    ``focus_offset`` shifts every sampled Z plane by a constant, in the same
    units as ``dz``; 0.0 (the default) applies no shift.
    """
    del illumination_na
    if not math.isfinite(focus_offset):
        raise ValueError(f"focus_offset must be finite, got {focus_offset!r}")
    detection_na = detection_na if detection_na is not None else na
    required_values = {
        "detection_na": detection_na,
        "wavelength": wavelength,
        "ni": ni,
        "ns": ns,
        "dxy": dxy,
        "dz": dz,
    }
    missing = [name for name, value in required_values.items() if value is None or value <= 0]
    if missing:
        raise ValueError(
            "Missing required optical/acquisition parameter(s): " + ", ".join(missing)
        )

    requested_kwargs = {
        "z": psf_size_z,
        "nx": psf_size_xy,
        "dz": dz,
        "dxy": dxy,
        "NA": detection_na,
        "wvl": wavelength,
        "ni": ni,
        "oversample_factor": oversample_factor,
        "model": psf_model,
    }
    if focus_offset != 0.0:
        # plane i is sampled at defocus (i - (psf_size_z - 1) / 2) * dz
        # + focus_offset -- the same psfmodels.tot_psf x_offset convention
        # (_centered_zv's own pz shift, generalized off of the integer-z
        # path). psfmodels emits a UserWarning "dz is ignored" whenever dz
        # is passed alongside a z sequence, so drop it here.
        lim = (psf_size_z - 1) * dz / 2
        requested_kwargs["z"] = np.linspace(-lim + focus_offset, lim + focus_offset, psf_size_z)
        del requested_kwargs["dz"]
    optional_kwargs = {
        "ns": ns,
        "ni0": ni0,
        "tg": tg,
        "tg0": tg0,
        "ng": ng,
        "ng0": ng0,
        "ti0": ti0,
    }
    requested_kwargs.update(
        {name: value for name, value in optional_kwargs.items() if value is not None}
    )
    signature = inspect.signature(pm.make_psf)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_kwargs:
        missing_params = [
            name for name in requested_kwargs if name not in signature.parameters
        ]
        if missing_params:
            raise RuntimeError(
                "psfmodels.make_psf API mismatch; missing expected parameter(s): "
                + ", ".join(missing_params)
            )

    psf = pm.make_psf(**requested_kwargs).astype(np.float32)
    return normalise_psf(np.maximum(psf - background, 0))


def _gaussian_pupil_amplitude(s: np.ndarray, fill_factor: float) -> np.ndarray:
    """Gaussian-apodized pupil field amplitude.

    The field amplitude falls to ``1/e**2`` of its on-axis value at
    ``s = fill_factor`` -- this is the field **amplitude**, not the
    intensity. The intensity-``1/e**2`` reading ``exp(-(s / fill_factor)**2)``
    (the conventional laser-beam-waist convention) is deliberately not used
    here.
    """
    return np.exp(-2.0 * (s / fill_factor) ** 2)


def _cylindrical_theta_interval_count(
    *,
    theta_max: float,
    k: float,
    x_max: float,
    max_abs_z: float,
    illumination_na: float,
    illumination_fill_factor: float | None,
) -> int:
    """Return the even Simpson interval count for the 1-D cylindrical integral."""
    s_samples = _APODIZED_THETA_SAMPLES_PER_HALF_CYCLE
    # The lateral and axial phase swings share one integrand here, so they
    # can add and are counted together rather than as separate candidates.
    x_swings = math.floor(theta_max * k * x_max / math.pi)
    z_swings = math.floor(k * max_abs_z * (1 - math.cos(theta_max)) / math.pi)
    candidates = [
        float(_APODIZED_MIN_THETA_INTERVALS),
        s_samples * (1 + x_swings + z_swings),
    ]
    if illumination_fill_factor is not None:
        candidates.append(math.ceil(4.0 / illumination_fill_factor))
    n_intervals = int(max(candidates))
    if n_intervals % 2 != 0:
        n_intervals += 1
    if n_intervals > _APODIZED_MAX_THETA_INTERVALS:
        raise ValueError(
            f"generate_cylindrical_illumination_psf requires {n_intervals} theta "
            f"quadrature intervals, exceeding the cap of {_APODIZED_MAX_THETA_INTERVALS} "
            f"-- the quadrature cannot resolve illumination_na={illumination_na!r}, "
            f"illumination_fill_factor={illumination_fill_factor!r} over a z extent "
            f"of {2.0 * max_abs_z!r}"
        )
    return n_intervals


def _cylindrical_line_intensity(
    *,
    illumination_na: float,
    illumination_fill_factor: float | None,
    wavelength: float,
    ni: float,
    dxy: float,
    zv: np.ndarray,
    x_sub: np.ndarray,
    sf: int,
    psf_size_xy: int,
) -> np.ndarray:
    """Return the pixel-averaged ``(len(zv), psf_size_xy)`` 1-D line intensity."""
    k = 2.0 * math.pi * ni / wavelength
    theta_max = math.asin(illumination_na / ni)
    n_intervals = _cylindrical_theta_interval_count(
        theta_max=theta_max,
        k=k,
        x_max=float(np.max(np.abs(x_sub))),
        max_abs_z=float(np.max(np.abs(zv))),
        illumination_na=illumination_na,
        illumination_fill_factor=illumination_fill_factor,
    )

    theta = np.linspace(0.0, theta_max, n_intervals + 1)
    simpson_weights = np.ones(n_intervals + 1, dtype=np.float64)
    simpson_weights[1:n_intervals:2] = 4.0
    simpson_weights[2:n_intervals:2] = 2.0
    simpson_weights *= (theta_max / n_intervals) / 3.0

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    if illumination_fill_factor is None:
        amplitude = np.ones_like(theta)
    else:
        amplitude = _gaussian_pupil_amplitude(
            sin_theta / math.sin(theta_max), illumination_fill_factor
        )
    weighted = simpson_weights * amplitude * np.sqrt(cos_theta)  # (n_nodes,)
    kernel = np.cos(k * sin_theta[:, None] * x_sub[None, :]) * weighted[:, None]

    # Chunk the (z x nodes) phase matrix if its complex128 footprint would
    # exceed ~256 MB.
    chunk_budget_bytes = 256 * 1024 * 1024
    rows_per_chunk = max(1, int(chunk_budget_bytes // (theta.size * 16)))
    field = np.empty((zv.size, x_sub.size), dtype=np.complex128)
    for start in range(0, zv.size, rows_per_chunk):
        stop = min(start + rows_per_chunk, zv.size)
        phase = np.exp(1j * k * zv[start:stop, None] * cos_theta[None, :])
        field[start:stop, :] = phase @ kernel

    intensity_sub = np.abs(field) ** 2  # (n_z, psf_size_xy * sf)
    return intensity_sub.reshape(zv.size, psf_size_xy, sf).mean(axis=2)


def generate_cylindrical_illumination_psf(
    *,
    illumination_na: float,
    wavelength: float,
    ni: float,
    dxy: float,
    dz: float,
    psf_size_z: int = 61,
    psf_size_xy: int = 128,
    oversample_factor: int = 3,
    background: float = 0.0,
    focus_offset: float = 0.0,
    illumination_fill_factor: float | None = None,
) -> np.ndarray:
    """Scalar 1-D Debye illumination PSF of an ideal cylindrical lens.

    In the pre-rotation frame (axis 0 propagation z, axis 1 y, axis 2 x) the
    field is ``E(x, z) = 2 * integral over 0..theta_max of A(s) sqrt(cos theta)
    cos(k x sin theta) exp(i k z cos theta) d theta`` with ``k = 2 pi ni /
    wavelength``. The integrand is even in theta, so the symmetric fan is twice
    the half fan (the factor 2 cancels in normalisation). A 1-D pupil has no
    ``sin theta`` area factor. The ``sqrt(cos theta)`` obliquity factor is an
    assumption.

    The beam is focused along pre-rotation X and uniform along pre-rotation Y:
    an ideal cylindrical lens with no finite sheet height. The aperture is hard
    (``A = 1``) with its edge at ``sin(theta_max) = illumination_na / ni``, so
    "illumination NA" keeps its aperture meaning. ``illumination_fill_factor``
    applies the 1-D Gaussian apodization ``A(s) = exp(-2 (s / f)**2)``,
    ``s = sin(theta) / sin(theta_max)``, a field amplitude as in
    ``_gaussian_pupil_amplitude``.

    This is an aberration-free scalar integral in a homogeneous medium of index
    ``ni``; the aberration parameters and ``psf_model`` do not apply.
    """
    for name, value in (
        ("illumination_na", illumination_na),
        ("wavelength", wavelength),
        ("ni", ni),
        ("dxy", dxy),
        ("dz", dz),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    if illumination_na >= ni:
        raise ValueError(
            "illumination_na must be < ni, got "
            f"illumination_na={illumination_na!r}, ni={ni!r}"
        )
    if not isinstance(psf_size_z, int) or psf_size_z < 1:
        raise ValueError(f"psf_size_z must be an int >= 1, got {psf_size_z!r}")
    if not isinstance(psf_size_xy, int) or psf_size_xy < 1:
        raise ValueError(f"psf_size_xy must be an int >= 1, got {psf_size_xy!r}")
    if not isinstance(oversample_factor, int) or oversample_factor < 1:
        raise ValueError(
            f"oversample_factor must be an int >= 1, got {oversample_factor!r}"
        )
    if not math.isfinite(background):
        raise ValueError(f"background must be finite, got {background!r}")
    if not math.isfinite(focus_offset):
        raise ValueError(f"focus_offset must be finite, got {focus_offset!r}")
    if illumination_fill_factor is not None and (
        not math.isfinite(illumination_fill_factor) or illumination_fill_factor <= 0
    ):
        raise ValueError(
            "illumination_fill_factor must be None or finite and > 0, got "
            f"{illumination_fill_factor!r}"
        )

    sf = oversample_factor
    lim = (psf_size_z - 1) * dz / 2.0
    zv = np.linspace(-lim + focus_offset, lim + focus_offset, psf_size_z)

    # psfmodels' beam-axis pixel coordinate.
    c = (((psf_size_xy * sf - 1) // 2) - (sf - 1) / 2.0) / sf
    idx = np.arange(psf_size_xy, dtype=np.float64)
    sub_offsets = (np.arange(sf, dtype=np.float64) + 0.5) / sf - 0.5
    x_sub = ((idx[:, None] - c + sub_offsets[None, :]) * dxy).ravel()

    line = _cylindrical_line_intensity(
        illumination_na=illumination_na,
        illumination_fill_factor=illumination_fill_factor,
        wavelength=wavelength,
        ni=ni,
        dxy=dxy,
        zv=zv,
        x_sub=x_sub,
        sf=sf,
        psf_size_xy=psf_size_xy,
    )
    psf = np.broadcast_to(
        line[:, None, :], (psf_size_z, psf_size_xy, psf_size_xy)
    ).copy()

    if not np.isfinite(psf).all() or float(psf.max()) <= 0.0:
        raise ValueError(
            "cylindrical illumination has no positive finite energy "
            f"(illumination_fill_factor={illumination_fill_factor!r}, "
            f"illumination_na={illumination_na!r})"
        )

    psf = (psf / psf.max()).astype(np.float32)
    return normalise_psf(np.maximum(psf - background, 0))


def _center_crop_or_pad(volume: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    output = np.zeros(shape, dtype=volume.dtype)
    source_slices = []
    dest_slices = []
    for current, target in zip(volume.shape, shape):
        if current >= target:
            source_start = (current - target) // 2
            dest_start = 0
            length = target
        else:
            source_start = 0
            dest_start = (target - current) // 2
            length = current
        source_slices.append(slice(source_start, source_start + length))
        dest_slices.append(slice(dest_start, dest_start + length))
    output[tuple(dest_slices)] = volume[tuple(source_slices)]
    return output


def _rotation_matrix(polar_deg: float, azimuthal_deg: float) -> np.ndarray:
    """Build the (Z, Y, X)-ordered rotation matrix ``Rz(azimuthal) @ Ry(polar)``."""
    # polar_deg is measured from the pre-rotation +Z propagation axis;
    # azimuthal_deg is measured from +X in the X-Y plane -- the standard
    # physics spherical convention, embedded into the (Z, Y, X) array index
    # order.
    # Both angles are in degrees, not radians.
    if not math.isfinite(polar_deg):
        raise ValueError(f"polar_deg must be finite, got {polar_deg!r}")
    if not math.isfinite(azimuthal_deg):
        raise ValueError(f"azimuthal_deg must be finite, got {azimuthal_deg!r}")
    theta = np.radians(polar_deg)
    phi = np.radians(azimuthal_deg)
    ry = np.array(
        [
            [np.cos(theta), 0.0, -np.sin(theta)],
            [0.0, 1.0, 0.0],
            [np.sin(theta), 0.0, np.cos(theta)],
        ]
    )
    rz = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi), np.sin(phi)],
            [0.0, -np.sin(phi), np.cos(phi)],
        ]
    )
    return rz @ ry


def _spherical_direction(polar_deg: float, azimuthal_deg: float) -> np.ndarray:
    """Return the (Z, Y, X) propagation unit vector for (polar_deg, azimuthal_deg)."""
    # (90.0, 0.0) is the fixed reference point mapping to the broadside
    # default, yielding (Z=0, Y=0, X=1). This is the single source of truth
    # for the direction vector -- callers never re-derive the trig themselves.
    return _rotation_matrix(polar_deg, azimuthal_deg) @ np.array([1.0, 0.0, 0.0])


def _match_legacy_cardinal(direction: np.ndarray) -> int | None:
    """Return the numpy.rot90 ``k`` for a cardinal direction, else None."""
    # Matches on the direction vector, not the (polar_deg, azimuthal_deg) pair,
    # so pole degeneracy at polar_deg 0/180 needs no special-case branch.
    # RIGHT_ANGLE_TOLERANCE is the direction-space tolerance for the match.
    legacy_directions = (
        (np.array([1.0, 0.0, 0.0]), 0),
        (np.array([0.0, 0.0, 1.0]), 1),
        (np.array([-1.0, 0.0, 0.0]), 2),
        (np.array([0.0, 0.0, -1.0]), 3),
    )
    for legacy_direction, k in legacy_directions:
        if np.max(np.abs(direction - legacy_direction)) <= RIGHT_ANGLE_TOLERANCE:
            return k
    return None


def _legacy_rot90_rotation(illumination: np.ndarray, k: int) -> np.ndarray:
    """Rotate by whole quarter turns with numpy.rot90 about axes (0, 2), then centre-crop or pad to the input shape."""
    rotated = np.rot90(illumination, k=k, axes=(0, 2))
    return _center_crop_or_pad(rotated, illumination.shape)


def rotate_illumination(
    illumination: np.ndarray,
    *,
    polar_deg: float,
    azimuthal_deg: float,
    dxy: float,
    dz: float,
) -> np.ndarray:
    """Rotate the pre-rotation illumination PSF to the requested cardinal direction.

    Uses a numpy.rot90 axis relabel for the four cardinal directions. Any
    other direction raises ValueError.
    """
    rotation = _rotation_matrix(polar_deg, azimuthal_deg)
    direction = rotation @ np.array([1.0, 0.0, 0.0])
    k = _match_legacy_cardinal(direction)
    if k is not None:
        return _legacy_rot90_rotation(illumination, k)
    raise ValueError(
        f"rotate_illumination: polar_deg={polar_deg}, azimuthal_deg={azimuthal_deg} is not "
        "one of the four cardinal directions this script supports"
    )


def _resolve_slit_fwhm(
    slit_width: float | None, slit_width_px: int | None, dz: float | None
) -> float:
    """Resolve the ASLM slit gate's FWHM in physical units from whichever form was supplied."""
    provided = [value for value in (slit_width, slit_width_px) if value is not None]
    if len(provided) != 1:
        raise ValueError(
            "Exactly one of slit_width or slit_width_px must be provided for psf_mode='aslm'"
        )
    if slit_width is not None:
        if slit_width <= 0:
            raise ValueError(f"slit_width must be > 0, got {slit_width!r}")
        return slit_width
    if slit_width_px <= 0:
        raise ValueError(f"slit_width_px must be > 0, got {slit_width_px!r}")
    # This conversion uses dz, the sample spacing of the integration axis
    # (pre-rotation axis 0).
    # This helper runs before generate_theoretical_psf's required_values
    # check, so it must validate every operand it uses arithmetically -- a
    # missing or non-positive dz would otherwise crash with an unguarded
    # TypeError instead of the module's normal descriptive ValueError.
    if dz is None or not math.isfinite(dz) or dz <= 0:
        raise ValueError(
            "dz must be finite and > 0 to convert slit_width_px to physical "
            f"units for psf_mode='aslm', got {dz!r}"
        )
    return slit_width_px * dz


def _gaussian_slit_window(size: int, fwhm: float, pixel_size: float) -> np.ndarray | None:
    """Build a geometric-midpoint-centered Gaussian taper, or None to skip the gate."""
    # Below one pixel_size sample, the window would resolve to a spike
    # narrower than the simulated grid can represent -- fall back to no
    # convolution instead (the waist-limited limit, identical to
    # light_sheet). No full-extent shortcut -- a very wide slit still
    # builds a real (near-uniform) window here and converges to the
    # swept-average sheet through the convolution itself, rather than
    # returning None early for an exact light_sheet reduction.
    if fwhm < pixel_size:
        return None
    sigma_px = (fwhm / pixel_size) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    center = (size - 1) / 2.0  # geometric midpoint, matches psfmodels' centered beam waist
    idx = np.arange(size, dtype=np.float64)
    window = np.exp(-0.5 * ((idx - center) / sigma_px) ** 2)
    return window.astype(np.float32)


def _apply_aslm_slit_gate(
    illumination: np.ndarray, fwhm: float, dz: float
) -> np.ndarray:
    """Convolve the pre-rotation illumination with a sweep-integrated slit window.

    The rolling shutter is assumed perfectly synchronized to the swept beam
    waist, so the effective illumination at any point along the beam's
    propagation axis (pre-rotation axis 0) is the raw illumination
    convolved with the Gaussian slit window -- not a static per-axis
    multiply. This runs before rotation, so it stays correct for any
    (polar_deg, azimuthal_deg) direction with no axis snapping.
    """
    n = illumination.shape[0]
    # The odd 2n-1 kernel length puts the centre tap exactly at index n - 1
    # for both odd and even n, so a same-mode convolution introduces no
    # shift, and the waist-offset integral spans the whole simulated window.
    window = _gaussian_slit_window(2 * n - 1, fwhm, dz)
    if window is None:
        gated = illumination  # below one dz sample, the waist-limited limit
    else:
        # Beam energy beyond the simulated window is treated as zero
        # (same-mode convolution, zero-padded edges); the clip removes tiny
        # negative values introduced by FFT round-off.
        gated = np.clip(
            fftconvolve(illumination, window.reshape(-1, 1, 1), mode="same", axes=0),
            0.0,
            None,
        ).astype(np.float32)

    # normalise_psf returns a zero-sum array unchanged and without error, so
    # without this guard a degenerate illumination input or a user-chosen
    # slit_width could produce an all-zero PSF seed that looks like a valid
    # one. This guard covers the no-convolution fallback path too.
    # Compare against the ungated illumination's own sum (a relative
    # statement), not a fixed constant.
    original_sum = float(np.abs(illumination).sum(dtype=np.float64))
    epsilon = max(float(np.finfo(np.float32).eps), original_sum * 1e-7)
    gated_sum = float(gated.sum(dtype=np.float64))
    if not np.isfinite(gated).all() or gated_sum <= epsilon:
        raise ValueError(
            f"ASLM slit-integrated illumination has no positive finite energy "
            f"(slit_width={fwhm!r}, dz={dz!r}); the sweep-integrated model "
            "assumes the rolling shutter is perfectly synchronized to the "
            "swept beam waist, so check the illumination parameters and "
            "slit_width rather than timing"
        )
    return gated


def generate_psf_seed(
    *,
    psf_mode: str,
    na: float,
    detection_na: float | None,
    illumination_na: float | None,
    wavelength: float,
    ni: float,
    ns: float | None,
    ni0: float | None,
    tg: float | None,
    tg0: float | None,
    ng: float | None,
    ng0: float | None,
    ti0: float | None,
    oversample_factor: int,
    psf_model: str,
    dxy: float,
    dz: float,
    psf_size_z: int,
    psf_size_xy: int,
    background: float,
    polar_deg: float = 90.0,
    azimuthal_deg: float = 0.0,
    slit_width: float | None = None,
    slit_width_px: int | None = None,
    emitter_offset: float = 0.0,
    illumination_fill_factor: float | None = None,
    illumination_beam: str = "pencil",
) -> np.ndarray:
    """Create a single-detection, light-sheet, or ASLM blind-estimation seed PSF.

    ASLM mode convolves the illumination PSF, in its pre-rotation frame, with
    a Gaussian slit window (FWHM ``slit_width``) along the beam's
    propagation axis (pre-rotation axis 0), before rotation. This is the
    sweep-integrated effective illumination seen by a rolling shutter
    assumed to be perfectly synchronized with the swept beam waist: the
    illuminated slit always sits exactly at the waist as it sweeps across
    the whole simulated window.

    A slit narrower than one dz sample applies no convolution -- the
    waist-limited limit, identical to ``light_sheet``. A very wide slit
    converges to the swept-average sheet rather than returning an exact
    ``light_sheet`` seed. Timing jitter, shutter/beam desynchronization, and
    sweep-velocity error are not modelled.

    ``emitter_offset`` is the emitter's offset from the illumination
    waist along the beam propagation direction, in the same units as
    ``dxy``/``dz``. A positive value moves the waist to a smaller
    pre-rotation axis-0 index, matching ``psfmodels.tot_psf``'s ``x_offset``
    convention. It is applied in the pre-rotation frame, so it follows the
    true beam direction rather than a camera-snapped axis. ``aslm``
    is invariant to it: the swept waist tracks the emitter's slit
    across the whole simulated window under the perfectly synchronized
    rolling-shutter assumption, so the effective illumination relative to
    the emitter does not depend on where the emitter sits. ``single`` has no
    illumination beam and requires 0.0.

    ``illumination_fill_factor`` is an opt-in, finite positive pupil
    apodization parameter for ``psf_mode='light_sheet'`` or ``'aslm'``: the
    illumination arm becomes a Gaussian-apodized scalar-Debye beam with field amplitude
    ``A(s) = exp(-2 (s / illumination_fill_factor)**2)``,
    ``s = sin(theta) / sin(theta_max)`` -- the field falls to ``1/e**2`` of
    its on-axis value at ``s = illumination_fill_factor`` (amplitude, not
    intensity). Leaving it ``None`` (the default) keeps the illumination arm
    the uniformly filled hard-aperture ``generate_theoretical_psf`` call. When
    set, the detection
    arm still uses ``psf_model`` and the full aberration parameter set
    (``ns``, ``ni0``, ``tg``, ``tg0``, ``ng``, ``ng0``, ``ti0``), but those
    parameters stop applying to the illumination arm, which becomes an
    aberration-free scalar Debye integral in a homogeneous medium of index
    ``ni``. This does not emit a warning when ``psf_model`` is left at its
    ``"vectorial"`` default, even though the illumination arm does not use
    it -- every fill-factor call would otherwise warn. ``single`` has no
    illumination beam and rejects a non-None value. In this script it is
    supported only with ``illumination_beam='cylindrical'``.

    ``illumination_beam`` selects the illumination arm for ``'light_sheet'`` or
    ``'aslm'``: ``"pencil"`` (the default) is the round-pupil beam;
    ``"cylindrical"`` is an ideal
    cylindrical-lens beam (``generate_cylindrical_illumination_psf``), a scalar
    1-D Debye integral focused along pre-rotation X and uniform along
    pre-rotation Y with no finite sheet height. The rotation puts the
    focused axis on detection Z at ``polar_deg=90`` for every azimuth. The
    detection arm keeps ``psf_model``; the aberration parameters stop applying
    to the illumination arm. ``illumination_fill_factor`` composes with it as a
    1-D Gaussian apodization. ``single`` rejects any value other than
    ``"pencil"``.
    """
    if psf_mode not in ("single", "light_sheet", "aslm"):
        raise ValueError(
            f"Unsupported psf_mode={psf_mode!r}; expected one of 'single', 'light_sheet', 'aslm'"
        )

    # Validate before generating any PSF, mirroring the angle validation
    # below.
    if not math.isfinite(emitter_offset):
        raise ValueError(f"emitter_offset must be finite, got {emitter_offset!r}")
    if psf_mode == "single" and emitter_offset != 0.0:
        raise ValueError(
            "emitter_offset applies only to psf_mode='light_sheet' or 'aslm' "
            f"(single has no illumination beam), got emitter_offset={emitter_offset!r}"
        )

    # Validate before generating any PSF, mirroring the emitter_offset guard
    # above.
    resolved_illumination_na: float | None = None
    if illumination_fill_factor is not None:
        if not math.isfinite(illumination_fill_factor) or illumination_fill_factor <= 0:
            raise ValueError(
                "illumination_fill_factor must be finite and > 0, got "
                f"{illumination_fill_factor!r}"
            )
        if psf_mode == "single":
            raise ValueError(
                "illumination_fill_factor applies only to psf_mode='light_sheet' or "
                "'aslm' (single has no illumination beam), got "
                f"illumination_fill_factor={illumination_fill_factor!r}"
            )
        if not isinstance(oversample_factor, int) or oversample_factor < 1:
            raise ValueError(
                "oversample_factor must be an int >= 1 when illumination_fill_factor "
                f"is set, got {oversample_factor!r}"
            )
        # Same fallback chain as the illumination call below:
        # illumination_na, else detection_na, else na.
        resolved_illumination_na = (
            illumination_na
            if illumination_na is not None
            else detection_na if detection_na is not None else na
        )
        # ni defaults to None on the CLI; compare it only after it is known to be
        # a usable number so a missing value is a ValueError, not a TypeError.
        if ni is None or not math.isfinite(ni) or ni <= 0:
            raise ValueError(
                "illumination_fill_factor requires a finite ni > 0, got "
                f"ni={ni!r}"
            )
        if (
            resolved_illumination_na is None
            or not math.isfinite(resolved_illumination_na)
            or resolved_illumination_na <= 0
            or resolved_illumination_na >= ni
        ):
            raise ValueError(
                "illumination_fill_factor requires a resolved illumination NA that "
                "is finite, > 0 and < ni; got illumination_na="
                f"{illumination_na!r}, detection_na={detection_na!r}, na={na!r}, "
                f"resolved={resolved_illumination_na!r}, ni={ni!r}"
            )

    # Validate the beam choice before generating any PSF.
    if illumination_beam not in _ILLUMINATION_BEAMS:
        raise ValueError(
            "illumination_beam must be one of 'pencil', 'cylindrical', got "
            f"{illumination_beam!r}"
        )
    if psf_mode == "single" and illumination_beam != "pencil":
        raise ValueError(
            "illumination_beam applies only to psf_mode='light_sheet' or 'aslm' "
            f"(single has no illumination beam), got illumination_beam={illumination_beam!r}"
        )
    if illumination_beam == "cylindrical":
        if not isinstance(oversample_factor, int) or oversample_factor < 1:
            raise ValueError(
                "oversample_factor must be an int >= 1 when illumination_beam="
                f"'cylindrical', got {oversample_factor!r}"
            )
        resolved_illumination_na = (
            illumination_na
            if illumination_na is not None
            else detection_na if detection_na is not None else na
        )
        if ni is None or not math.isfinite(ni) or ni <= 0:
            raise ValueError(
                "illumination_beam='cylindrical' requires a finite ni > 0, got "
                f"ni={ni!r}"
            )
        if (
            resolved_illumination_na is None
            or not math.isfinite(resolved_illumination_na)
            or resolved_illumination_na <= 0
            or resolved_illumination_na >= ni
        ):
            raise ValueError(
                "illumination_beam='cylindrical' requires a resolved illumination NA "
                "that is finite, > 0 and < ni; got illumination_na="
                f"{illumination_na!r}, detection_na={detection_na!r}, na={na!r}, "
                f"resolved={resolved_illumination_na!r}, ni={ni!r}"
            )

    # Validate before generating any PSF, including the non-finite-angle check
    # -- a NaN/inf polar_deg or azimuthal_deg must fail loudly rather than flow
    # through normalise_psf's nan_to_num into an all-zero seed. The direction
    # vector itself is not consumed here; the call is made only for its
    # validation side effect.
    if psf_mode != "single":
        _spherical_direction(polar_deg, azimuthal_deg)

    slit_fwhm: float | None = None
    if psf_mode == "aslm":  # validate before generating any PSF
        slit_fwhm = _resolve_slit_fwhm(slit_width, slit_width_px, dz)

    detection = generate_theoretical_psf(
        na=na,
        detection_na=detection_na,
        illumination_na=illumination_na,
        wavelength=wavelength,
        ni=ni,
        ns=ns,
        ni0=ni0,
        tg=tg,
        tg0=tg0,
        ng=ng,
        ng0=ng0,
        ti0=ti0,
        oversample_factor=oversample_factor,
        psf_model=psf_model,
        dxy=dxy,
        dz=dz,
        psf_size_z=psf_size_z,
        psf_size_xy=psf_size_xy,
        background=background,
    )

    if psf_mode == "single":
        return normalise_psf(detection)

    # Only light_sheet forwards the off-waist emitter offset. The ASLM waist
    # is swept over the whole simulated window with the rolling-shutter slit
    # perfectly synchronized to the emitter's row, so the effective
    # illumination relative to the emitter does not depend on where the
    # emitter sits -- the ASLM seed is therefore intentionally invariant
    # to emitter_offset, and always forwards 0.0.
    illumination_focus_offset = emitter_offset if psf_mode == "light_sheet" else 0.0

    if illumination_beam == "cylindrical":
        # The 1-D Debye beam; the aberration parameters and psf_model do not
        # apply to it (see its docstring).
        illumination = generate_cylindrical_illumination_psf(
            illumination_na=resolved_illumination_na,
            wavelength=wavelength,
            ni=ni,
            dxy=dxy,
            dz=dz,
            psf_size_z=psf_size_z,
            psf_size_xy=psf_size_xy,
            oversample_factor=oversample_factor,
            background=background,
            focus_offset=illumination_focus_offset,
            illumination_fill_factor=illumination_fill_factor,
        )
    elif illumination_fill_factor is None:
        illumination = generate_theoretical_psf(
            na=na,
            detection_na=illumination_na if illumination_na is not None else detection_na,
            illumination_na=illumination_na,
            wavelength=wavelength,
            ni=ni,
            ns=ns,
            ni0=ni0,
            tg=tg,
            tg0=tg0,
            ng=ng,
            ng0=ng0,
            ti0=ti0,
            oversample_factor=oversample_factor,
            psf_model=psf_model,
            dxy=dxy,
            dz=dz,
            psf_size_z=psf_size_z,
            psf_size_xy=psf_size_xy,
            background=background,
            focus_offset=illumination_focus_offset,
        )
    else:
        raise ValueError(
            "illumination_fill_factor with illumination_beam='pencil' (the Gaussian-apodized "
            "pencil beam) is not supported in this script"
        )
    if psf_mode == "aslm":
        illumination = _apply_aslm_slit_gate(illumination, slit_fwhm, dz)
    rotated = rotate_illumination(
        illumination, polar_deg=polar_deg, azimuthal_deg=azimuthal_deg, dxy=dxy, dz=dz
    )
    return normalise_psf(detection * rotated)


if __name__ == "__main__":
    main()
