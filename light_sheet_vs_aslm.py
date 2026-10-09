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
"""Contrast static light_sheet and ASLM PSF seed generation via generate_psf_seed().

Notes / limitations (printed to stdout by main(), kept out of the figure
itself):
    - Illumination model: an ideal cylindrical-lens beam -- a scalar 1-D
      Debye integral focused along the detection axis (Z) and uniform along
      y, with no finite sheet height. The pupil is 1-D Gaussian-apodized at
      ILLUMINATION_FILL_FACTOR = 0.7, the same beam as
      resolution_vs_fov_by_na.py. The single (detection-only) seed has no
      illumination arm.
    - Sampling: cubic voxels (dz == dxy), ni0 == ni. slit_width is the FWHM
      of a Gaussian window; Dean et al. 2015 used a rectangular
      slit W = 2*xR, and a Gaussian with the same second moment has FWHM
      ~0.68*W.

Run it with `uv run light_sheet_vs_aslm.py`; the figure is written to
output/light_sheet_vs_aslm.png beside this file.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psfmodels as pm
from matplotlib.colors import LogNorm
from scipy.signal import fftconvolve

# Optical parameters for the demo. The two aperture entries below are
# deliberately raised for this demo.
#
# Parameter choices:
#   - "dz" == "dxy": the cardinal-direction rotation path
#     (`_legacy_rot90_rotation`) relabels pre-rotation axes onto the display
#     axes with no resampling. That relabel is exact only for cubic voxels.
#   - "psf_size_z" == "psf_size_xy" (a cube window): after the relabel,
#     pre-rotation axis 0 (the "psf_size_z" samples) lands on the display X
#     axis. An equal size fills X edge-to-edge with no zero-padded white
#     bands. The odd size (129) puts the focal plane exactly on a sample,
#     rather than between two samples at a 63.5-style half-integer centre.
#     The 13.9 um window (129 * 0.108 um) spans many Rayleigh ranges at
#     illumination_na 0.6 and still comfortably holds the OFF_WAIST_UM offset
#     below.
#   - "ni0" == "ni" (1.33): "ni0=None" falls back to psfmodels' own default
#     design immersion index (1.515, oil), producing a spherical-aberration
#     focal shift in these water-immersion demos.
COMMON = {
    "na": 1.0,
    "detection_na": 1.1,
    "illumination_na": 0.6,
    "wavelength": 0.561,
    "ni": 1.33,
    "ns": 1.33,
    "ni0": 1.33,
    "tg": None,
    "tg0": None,
    "ng": None,
    "ng0": None,
    "ti0": None,
    "oversample_factor": 3,
    "psf_model": "vectorial",
    "dxy": 0.108,
    "dz": 0.108,
    "psf_size_z": 129,
    "psf_size_xy": 129,
    "background": 0.0,
    "polar_deg": 90.0,
    "azimuthal_deg": 0.0,
}
# ASLM demo slit width: the FWHM of the Gaussian slit window, in um.
SLIT_WIDTH = 2.0

# Illumination beam for the light_sheet and aslm seeds. Kept out of COMMON
# because generate_psf_seed rejects the keyword for psf_mode="single".
ILLUMINATION_BEAM = "cylindrical"

# The 1-D Gaussian apodization of the cylindrical pupil,
# A(s) = exp(-2 (s / ILLUMINATION_FILL_FACTOR)^2) across the 1-D aperture.
ILLUMINATION_FILL_FACTOR = 0.7

# The emitter's offset from the illumination waist, along the propagation
# direction, for the off-waist light_sheet column. Roughly ten paraxial
# Rayleigh ranges at illumination_na 0.6 -- a sanity scale only, not a
# derivation. It stays well inside the COMMON window's +-6.97 um half-extent.
OFF_WAIST_UM = 5.0

# Display-only zoom for every MIP panel below: set_xlim/set_ylim crops each
# panel's *view* to this physical +-half-width, in micrometres -- the simulated
# window itself stays the full COMMON psf_size_z/psf_size_xy cube, nothing here
# changes what is simulated. At the per-panel LogNorm floor, the raw +-6.97 um
# simulated extent buries the in-focus core and the off-waist spread inside a
# handful of screen pixels.
MIP_CROP_HALF_WIDTH_UM = 3.0

# One fixed colour per curve, shared by every line panel below (the axial
# profile, the energy-sectioning panel, and the outside-core panel).
# matplotlib's default colour cycle assigns colours by plot ORDER within each
# axes, and the panels plot their curves in different orders; an explicit
# per-curve colour map removes that dependency on plot order.
CURVE_COLORS = {
    "light_sheet": "tab:blue",
    "light_sheet_off_waist": "tab:orange",
    "aslm": "tab:green",
}

# Illumination-model disclosure, printed to stdout by main().
ILLUMINATION_MODEL_NOTE = (
    "Illumination model: an ideal cylindrical-lens beam (scalar 1-D Debye "
    "integral), focused along the detection axis (Z) and uniform along y with "
    "no finite sheet height. The pupil is 1-D Gaussian-apodized at "
    "illumination_fill_factor = "
    f"{ILLUMINATION_FILL_FACTOR}, as in resolution_vs_fov_by_na.py. The "
    "single (detection-only) seed has no illumination arm."
)

# Sampling invariants this script enforces (_require_simulation_sampling) and
# the Gaussian-vs-Dean slit mapping, so a reader can trace slit_width back to
# Dean et al. 2015's rectangular convention without re-deriving it. A Gaussian
# window with the same second moment as Dean's rectangular W has
# sigma = W/sqrt(12) ~= 0.289*W, whose FWHM (2*sqrt(2*ln(2))*sigma) is ~0.68*W.
SAMPLING_AND_SLIT_NOTE = (
    "Sampling: cubic voxels (dz == dxy), ni0 == ni. slit_width is the FWHM "
    "of a Gaussian window; Dean et al. 2015 used a rectangular slit "
    "W = 2*xR, and a Gaussian with the same second moment has FWHM ~0.68*W."
)

# Relative tolerance (1e-6) for float32 noise when comparing adjacent samples.
# Each of this file's three crossing walks (axial_fwhm, lateral_fwhm,
# measured_illumination_cross_section_fwhm_um) has its own nested
# _bound_to_first_local_minimum closure that reads this constant. Each closure
# recognizes a flat-bottomed/tied-minimum plateau (a strict descent opens a
# candidate run, extended while adjacent samples tie within this tolerance,
# confirmed as the lobe-bounding minimum only by an above-threshold rise)
# and truncates inclusive of the LAST plateau sample.
_LOBE_MIN_RTOL = 1e-6

# The two failure-reason tags. "No local minimum" is the more fundamental
# failure and takes precedence when a crossing fails for different reasons on
# each side.
_NO_LOCAL_MINIMUM = "no local minimum before array edge"
_LOBE_TOO_NARROW = "central lobe too narrow to reach half-max"


def _require_simulation_sampling(common: dict) -> None:
    """Reject anisotropic sampling or a mismatched immersion index before simulating.

    The cardinal-direction rotation path (`_legacy_rot90_rotation`) is an
    exact pre-rotation-axis relabel only for cubic voxels; a mismatched `ni0`
    reintroduces a spherical-aberration focal shift.
    """
    dz = common["dz"]
    dxy = common["dxy"]
    if dz != dxy:
        raise ValueError(
            f"_require_simulation_sampling: dz ({dz}) != dxy ({dxy}); the rot90 "
            "rotation path is an exact axis relabel only for cubic voxels"
        )
    ni0 = common["ni0"]
    ni = common["ni"]
    if ni0 != ni:
        raise ValueError(
            f"_require_simulation_sampling: ni0 ({ni0}) != ni ({ni}); a mismatched ni0 "
            "reintroduces spherical aberration via psfmodels' own default design index"
        )


def describe_environment() -> None:
    """Print this script's resolved path and whether cupy is importable in this environment."""
    print(f"script: {Path(__file__).resolve()}")
    try:
        # Presence check only -- never import cupy itself.
        cupy_present = importlib.util.find_spec("cupy") is not None
    except Exception:
        cupy_present = False  # probe failed; treat as "not confirmed present"
    print(f"cupy present in this environment: {'yes' if cupy_present else 'no'}")


def summarise_mode(label: str, seed: np.ndarray, gate: str) -> None:
    """Print one shape/energy/gate summary line for a generated PSF seed."""
    energy = float(seed.sum())
    print(f"{label} shape={seed.shape} energy={energy:.6f} gate={gate}")


def axial_profile(seed: np.ndarray) -> np.ndarray:
    """Return the Z intensity profile of a (Z, Y, X) seed through its true peak.

    Locates the global peak with `np.unravel_index(np.argmax(seed), seed.shape)`
    rather than `shape // 2` -- the peak lands within one sample of the array
    centre for these cubic, ni0==ni parameters, but the exact index can still
    differ by a sample between light_sheet and aslm seeds.
    """
    peak_z, peak_y, peak_x = np.unravel_index(np.argmax(seed), seed.shape)
    return seed[:, peak_y, peak_x].astype(np.float64)


def lateral_fwhm(psf: np.ndarray, dxy: float) -> float:
    """Return the lateral (X) half-max width of a (Z, Y, X) PSF seed through its true peak.

    Locates the global peak the same way `axial_profile` does -- via
    `np.unravel_index(np.argmax(...))`, never a per-slice search, since a
    per-slice argmax drifts onto sidelobes far from focus. Walks outward
    along X from that peak row and linearly interpolates the half-maximum
    crossing on each side, using the same first-crossing interpolation as
    `axial_fwhm`. The outward walk on each side is bounded to the first
    flanking local minimum of the central lobe, computed independently of
    `axial_fwhm` and `measured_illumination_cross_section_fwhm_um`. Raises
    ValueError, rather than returning a sentinel, when the peak is
    non-positive or a crossing is not found within the bounded volume (the
    message names the failure reason) -- `main()` needs a definite float to
    build the core radius from.
    """
    peak_z, peak_y, peak_x = np.unravel_index(np.argmax(psf), psf.shape)
    profile = psf[peak_z, peak_y, :].astype(np.float64)
    peak_value = profile[peak_x]
    if peak_value <= 0:
        raise ValueError("lateral_fwhm: peak value is not positive, cannot measure half-max width")
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

    def _bound_to_first_local_minimum(indices: np.ndarray) -> tuple[np.ndarray, bool]:
        threshold = _LOBE_MIN_RTOL * peak_value
        for i in range(1, indices.size - 1):
            if profile[indices[i - 1]] > profile[indices[i]]:
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
                    return indices[: j + 1], True
        return indices, False

    left_indices = np.arange(peak_x, -1, -1)  # peak -> start, descending
    right_indices = np.arange(peak_x, profile.size)  # peak -> end, ascending

    left_bounded, left_found_minimum = _bound_to_first_local_minimum(left_indices)
    right_bounded, right_found_minimum = _bound_to_first_local_minimum(right_indices)

    left_crossing = _crossing(left_bounded)
    right_crossing = _crossing(right_bounded)
    if left_crossing is None or right_crossing is None:
        no_minimum_on_a_failed_side = (
            left_crossing is None and not left_found_minimum
        ) or (right_crossing is None and not right_found_minimum)
        reason = _NO_LOCAL_MINIMUM if no_minimum_on_a_failed_side else _LOBE_TOO_NARROW
        raise ValueError(
            f"lateral_fwhm: half-max crossing not found within the volume ({reason})"
        )
    return (right_crossing - left_crossing) * dxy


def outside_core_fraction(seed: np.ndarray, r_core_um: float, dxy: float) -> np.ndarray:
    """Per Z plane, the lateral energy fraction outside a fixed detection-core disc.

    The core is a disc of radius `r_core_um` centred on the seed's GLOBAL
    peak lateral position (`np.unravel_index(np.argmax(seed), seed.shape)`)
    -- fixed for every plane, never re-located per slice, matching
    `axial_profile`'s fixed-peak convention. The metric is
    `1 - E_core / E_plane` per Z. A plane carrying zero total energy maps to
    0.0 (there is no out-of-focus energy to report there) -- this panel's
    purpose is showing where energy is lost, not flagging degenerate planes.
    """
    _, peak_y, peak_x = np.unravel_index(np.argmax(seed), seed.shape)
    yy, xx = np.meshgrid(
        np.arange(seed.shape[1]), np.arange(seed.shape[2]), indexing="ij"
    )
    core_mask = np.hypot((yy - peak_y) * dxy, (xx - peak_x) * dxy) <= r_core_um
    e_plane = seed.sum(axis=(1, 2), dtype=np.float64)
    e_core = (seed * core_mask[None, :, :]).sum(axis=(1, 2), dtype=np.float64)
    fraction_out = np.zeros_like(e_plane)
    positive = e_plane > 0
    fraction_out[positive] = 1.0 - (e_core[positive] / e_plane[positive])
    return fraction_out


def axial_fwhm(psf: np.ndarray, dz: float) -> float:
    """Return the axial (Z) half-max width of a (Z, Y, X) PSF seed through its true peak.

    Mirrors `lateral_fwhm`'s first-crossing linear-interpolation convention,
    walked outward along Z instead of X from the seed's global peak
    (`np.unravel_index(np.argmax(...))`). `main()` measures this once
    on `seed_single` to get the DOF (depth of field) window that
    `in_focus_energy_fraction` and the energy-sectioning panel both use.
    The outward walk on each side is bounded to the first flanking local
    minimum of the central lobe, computed independently of `lateral_fwhm`
    and `measured_illumination_cross_section_fwhm_um`.
    Raises ValueError, rather than returning a sentinel, when the peak is
    non-positive or a crossing is not found within the bounded volume (the
    message names the failure reason).
    """
    peak_z, peak_y, peak_x = np.unravel_index(np.argmax(psf), psf.shape)
    profile = psf[:, peak_y, peak_x].astype(np.float64)
    peak_value = profile[peak_z]
    if peak_value <= 0:
        raise ValueError("axial_fwhm: peak value is not positive, cannot measure half-max width")
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

    def _bound_to_first_local_minimum(indices: np.ndarray) -> tuple[np.ndarray, bool]:
        threshold = _LOBE_MIN_RTOL * peak_value
        for i in range(1, indices.size - 1):
            if profile[indices[i - 1]] > profile[indices[i]]:
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
                    return indices[: j + 1], True
        return indices, False

    left_indices = np.arange(peak_z, -1, -1)  # peak -> start, descending
    right_indices = np.arange(peak_z, profile.size)  # peak -> end, ascending

    left_bounded, left_found_minimum = _bound_to_first_local_minimum(left_indices)
    right_bounded, right_found_minimum = _bound_to_first_local_minimum(right_indices)

    left_crossing = _crossing(left_bounded)
    right_crossing = _crossing(right_bounded)
    if left_crossing is None or right_crossing is None:
        no_minimum_on_a_failed_side = (
            left_crossing is None and not left_found_minimum
        ) or (right_crossing is None and not right_found_minimum)
        reason = _NO_LOCAL_MINIMUM if no_minimum_on_a_failed_side else _LOBE_TOO_NARROW
        raise ValueError(
            f"axial_fwhm: half-max crossing not found within the volume ({reason})"
        )
    return (right_crossing - left_crossing) * dz


def plane_energy_profile(seed: np.ndarray) -> np.ndarray:
    """Return the per-Z-plane energy fraction of a (Z, Y, X) seed, summing to 1.0.

    `seed.sum(axis=(1, 2), dtype=np.float64)` divided by the seed's total
    energy -- the raw per-plane energy vector `in_focus_energy_fraction`
    below windows and sums. Raises ValueError when the seed's total energy
    is not positive (there is nothing to normalise a fraction against).
    """
    plane_energy = seed.sum(axis=(1, 2), dtype=np.float64)
    total = float(plane_energy.sum())
    if total <= 0:
        raise ValueError("plane_energy_profile: seed total energy is not positive")
    return plane_energy / total


def in_focus_energy_fraction(seed: np.ndarray, half_window_um: float, dz: float) -> float:
    """Return the fraction of `seed`'s energy within `half_window_um` of the array centre plane.

    Measures optical sectioning around the emitter's focal plane: the sum of
    `plane_energy_profile` over planes whose offset from the centre plane
    index `(psf_size_z - 1) / 2` is at most `half_window_um`. Uses the array
    CENTRE, not each seed's own argmax plane, because every seed's peak lands
    within one sample of that centre under this script's cubic, ni0==ni
    COMMON -- a fixed reference keeps the light_sheet-vs-aslm comparison
    honest even where the peak drifts by a sample between modes.
    """
    profile = plane_energy_profile(seed)
    centre = (seed.shape[0] - 1) / 2.0
    offsets_um = (np.arange(seed.shape[0]) - centre) * dz
    in_window = np.abs(offsets_um) <= half_window_um
    return float(profile[in_window].sum())


def detection_core_radius_um(seed_single: np.ndarray, dxy: float) -> float:
    """Return the detection-core radius: half the detection lateral FWHM (a radius, not the full width)."""
    return lateral_fwhm(seed_single, dxy) / 2.0


def measured_illumination_cross_section_fwhm_um(
    seed_light_sheet_off_waist: np.ndarray, seed_single: np.ndarray, dz: float
) -> float:
    """Return the illumination sheet's half-max thickness along detection Z at the emitter's (Y, X).

    `generate_psf_seed` composes `light_sheet = normalise(detection *
    rotated_illumination)`, and `seed_single = normalise(detection)` is the
    same `detection` factor up to a positive scalar that cancels out of any
    ratio. `seed_light_sheet_off_waist / seed_single`, evaluated along the
    detection peak's own (Y, X) column, therefore recovers the illumination
    sheet's thickness profile along Z -- up to an irrelevant positive scale --
    at the (Y, X) where the fixed emitter actually sits: it is the
    illumination *waist* that `emitter_offset` moves off that plane, not the
    emitter itself (see `generate_psf_seed`'s docstring). Walking along
    display X instead would not work: at polar_deg=90, azimuthal_deg=0 display
    X is the propagation axis, so that walk would measure the beam's half-max
    length along propagation around the moved waist, and for the f = 0.7
    cylindrical beam it finds no flanking minimum or half-max crossing inside
    the window. This is a diagnostic figure annotation (computed from these
    two already-generated seeds) -- it is not the detection core radius or the
    in-focus energy fraction, and no other code depends on it being accurate
    away from this one column. The outward walk on each side is bounded to the
    first flanking local minimum of the central lobe, computed independently
    of `axial_fwhm` and `lateral_fwhm`.
    """
    _, peak_y, peak_x = np.unravel_index(np.argmax(seed_single), seed_single.shape)
    detection = seed_single[:, peak_y, peak_x].astype(np.float64)
    illuminated = seed_light_sheet_off_waist[:, peak_y, peak_x].astype(np.float64)
    epsilon = max(float(np.finfo(np.float64).eps), float(detection.max()) * 1e-6)
    ratio = illuminated / np.clip(detection, epsilon, None)

    peak_index = int(np.argmax(ratio))
    peak_value = ratio[peak_index]
    if peak_value <= 0:
        raise ValueError(
            "measured_illumination_cross_section_fwhm_um: recovered illumination ratio "
            "peak is not positive, cannot measure half-max width"
        )
    half_max = peak_value / 2.0

    def _crossing(indices: np.ndarray) -> float | None:
        values = ratio[indices]
        below = np.where(values < half_max)[0]
        if below.size == 0:
            return None
        edge = below[0]
        i0, i1 = indices[edge - 1], indices[edge]
        v0, v1 = ratio[i0], ratio[i1]
        frac = (half_max - v0) / (v1 - v0)
        return i0 + frac * (i1 - i0)

    def _bound_to_first_local_minimum(indices: np.ndarray) -> tuple[np.ndarray, bool]:
        threshold = _LOBE_MIN_RTOL * peak_value
        for i in range(1, indices.size - 1):
            if ratio[indices[i - 1]] > ratio[indices[i]]:
                j = i
                while (
                    j + 1 < indices.size
                    and abs(ratio[indices[j + 1]] - ratio[indices[j]]) <= threshold
                ):
                    j += 1
                if (
                    j + 1 < indices.size
                    and ratio[indices[j + 1]] - ratio[indices[j]] > threshold
                ):
                    return indices[: j + 1], True
        return indices, False

    left_indices = np.arange(peak_index, -1, -1)  # peak -> start, descending
    right_indices = np.arange(peak_index, ratio.size)  # peak -> end, ascending

    left_bounded, left_found_minimum = _bound_to_first_local_minimum(left_indices)
    right_bounded, right_found_minimum = _bound_to_first_local_minimum(right_indices)

    left_crossing = _crossing(left_bounded)
    right_crossing = _crossing(right_bounded)
    if left_crossing is None or right_crossing is None:
        no_minimum_on_a_failed_side = (
            left_crossing is None and not left_found_minimum
        ) or (right_crossing is None and not right_found_minimum)
        reason = _NO_LOCAL_MINIMUM if no_minimum_on_a_failed_side else _LOBE_TOO_NARROW
        raise ValueError(
            "measured_illumination_cross_section_fwhm_um: half-max crossing not found "
            f"within the volume ({reason})"
        )
    return (right_crossing - left_crossing) * dz


def build_comparison_figure(
    seed_light_sheet: np.ndarray,
    seed_light_sheet_off_waist: np.ndarray,
    seed_aslm: np.ndarray,
    common: dict,
    slit_width: float,
    *,
    off_waist_um: float,
    r_core_um: float,
    illumination_fwhm_um: float,
    dof_um: float,
) -> plt.Figure:
    """Assemble the light_sheet-vs-aslm comparison figure: MIPs, axial profile, sectioning, and outside-core panels.

    Every MIP panel is plotted in physical micrometres (derived from
    `dz`/`dxy`), centered on zero, cropped to `MIP_CROP_HALF_WIDTH_UM` on
    every axis. The three MIP columns are: light_sheet at the waist,
    light_sheet with the emitter `off_waist_um` off-waist, and aslm -- aslm
    gets only one column because it is bit-identical for every
    `emitter_offset`. There is no detection-only column; the sheet thickness
    measured from `seed_single` is the `illumination_fwhm_um` annotation,
    computed by the caller, not plotted as its own panel. Each panel is
    normalised to its *own* peak (not a shared per-row max) with a fixed
    LogNorm floor, so the off-waist and waist tails are both visible at once
    instead of one clipping the other; `illumination_fwhm_um` (see
    `measured_illumination_cross_section_fwhm_um`) annotates the off-waist
    column with the measured illumination thickness at the emitter's plane.
    `r_core_um` (see `detection_core_radius_um`) sets the fixed
    detection-core radius for the bottom outside-core-fraction panel
    (see `outside_core_fraction`). `dof_um` (see `axial_fwhm`) sets the
    depth-of-field half-window for the energy-sectioning panel (see
    `in_focus_energy_fraction`). Every line panel below plots its three
    curves in the same `CURVE_COLORS` order, so a curve's colour is
    identical across every panel it appears in.
    """
    dz = common["dz"]
    dxy = common["dxy"]
    psf_size_z = common["psf_size_z"]
    psf_size_xy = common["psf_size_xy"]
    z_extent_um = psf_size_z * dz
    lateral_extent_um = psf_size_xy * dxy

    # XZ MIP: max over Y (axis=1) -> (Z, X). YZ MIP: max over X (axis=2) -> (Z, Y).
    xz_light_sheet = seed_light_sheet.max(axis=1)
    xz_light_sheet_off_waist = seed_light_sheet_off_waist.max(axis=1)
    xz_aslm = seed_aslm.max(axis=1)
    yz_light_sheet = seed_light_sheet.max(axis=2)
    yz_light_sheet_off_waist = seed_light_sheet_off_waist.max(axis=2)
    yz_aslm = seed_aslm.max(axis=2)

    # Extent centered on zero (left, right, bottom, top) with origin='upper':
    # index 0 (top row) -> -z_extent/2, last row -> +z_extent/2. Every column
    # and both projections share this same lateral/axial physical framing so
    # the focal plane annotated below lands exactly at Z=0.
    xz_extent = (
        -lateral_extent_um / 2,
        lateral_extent_um / 2,
        z_extent_um / 2,
        -z_extent_um / 2,
    )
    yz_extent = xz_extent  # same physical Z/lateral spans for both projections

    # Each MIP panel is normalised to its OWN peak (not a shared per-row max)
    # with a fixed LogNorm floor. The off-waist and waist/aslm panels differ
    # by roughly an order of magnitude in peak brightness (off-waist spreads
    # the same total energy over a much larger volume), so a shared scale
    # would clip everything but the brightest panel's tails into the LogNorm
    # floor. Because every panel is independently peak-normalised, the
    # shared colourbar below is labelled "intensity / panel peak", not an
    # absolute radiometric scale.
    # The floor is 1e-3: under the cylindrical beam the 1e-3..1e-4 band is a
    # featureless halo of the detection PSF's lateral tails that washes the
    # space around each core.
    panel_norm = LogNorm(vmin=1e-3, vmax=1.0)

    def _peak_normalised(panel: np.ndarray) -> np.ndarray:
        peak = float(panel.max())
        if peak <= 0:
            peak = float(np.finfo(np.float64).eps)
        return panel / peak

    # constrained_layout (not tight_layout): tight_layout does not account for
    # the per-row colorbars added below and produces overlapping panels/text.
    # 5-row gridspec: rows 0-1 are the 2x3 MIP grid (light_sheet waist,
    # light_sheet off-waist, aslm), row 2 spans all three columns for the
    # axial intensity-profile overlay, row 3 spans all three columns for the
    # plane-energy optical-sectioning panel, and row 4 spans all three
    # columns for the outside-core-fraction panel.
    fig = plt.figure(figsize=(15, 20), constrained_layout=True)
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 0.8, 0.8, 0.8])
    axes = np.array(
        [
            [fig.add_subplot(gs[0, col]) for col in range(3)],
            [fig.add_subplot(gs[1, col]) for col in range(3)],
        ]
    )

    row_specs = (
        (
            axes[0, :],
            (xz_light_sheet, xz_light_sheet_off_waist, xz_aslm),
            xz_extent,
            "XZ",
            "X",
        ),
        (
            axes[1, :],
            (yz_light_sheet, yz_light_sheet_off_waist, yz_aslm),
            yz_extent,
            "YZ",
            "Y",
        ),
    )

    for ax_row, panels, extent, proj_label, lateral_label in row_specs:
        ax_waist, ax_off_waist, ax_aslm = ax_row
        panel_waist, panel_off_waist, panel_aslm = panels

        # aspect="equal" (not "auto"): voxels are cubic (dz == dxy), so an
        # equal-aspect, index-based panel already matches the displayed
        # geometry to the physical micrometre extent -- no visual stretch
        # correction is needed here.
        ax_waist.imshow(
            _peak_normalised(panel_waist), aspect="equal", norm=panel_norm, extent=extent
        )
        ax_waist.set_title(f"light sheet {proj_label} MIP\nemitter at the waist", fontsize=8)

        ax_off_waist.imshow(
            _peak_normalised(panel_off_waist), aspect="equal", norm=panel_norm, extent=extent
        )
        ax_off_waist.set_title(
            f"light sheet {proj_label} MIP\nemitter {off_waist_um:.1f} um off-waist\n"
            f"sheet FWHM {illumination_fwhm_um:.1f} um at this position",
            fontsize=8,
        )

        im_aslm = ax_aslm.imshow(
            _peak_normalised(panel_aslm), aspect="equal", norm=panel_norm, extent=extent
        )
        # Three short lines, not two long ones: a single long second line
        # ran off the right edge of the figure at fontsize 9.
        ax_aslm.set_title(
            f"ASLM {proj_label} MIP\n"
            f"slit width = {slit_width:.2f} um, integrated along propagation\n"
            "(X at this direction), identical at every FOV position",
            fontsize=8,
        )

        for ax in (ax_waist, ax_off_waist, ax_aslm):
            ax.set_xlabel(f"{lateral_label} (um)")
            ax.set_ylabel("Z (um)")
            # Display-only crop (MIP_CROP_HALF_WIDTH_UM) -- the simulated
            # window itself is unchanged (still the full COMMON psf_size
            # cube); only the plotted view is zoomed, so the in-focus core
            # and the off-waist spread are both visible at once instead of
            # squeezed into a few pixels of the full +-6.97 um extent.
            ax.set_xlim(-MIP_CROP_HALF_WIDTH_UM, MIP_CROP_HALF_WIDTH_UM)
            ax.set_ylim(MIP_CROP_HALF_WIDTH_UM, -MIP_CROP_HALF_WIDTH_UM)

        cbar = fig.colorbar(im_aslm, ax=list(ax_row), shrink=0.85, pad=0.02)
        cbar.set_label("intensity / panel peak (log)")

    # Axial intensity-profile overlay: makes the waist / off-waist / aslm
    # difference readable without the reader computing anything. Z
    # coordinate centered on zero via pixel centers, the same "centered on
    # zero" convention the projection panels above use for their extent.
    profile_light_sheet = axial_profile(seed_light_sheet)
    profile_light_sheet_off_waist = axial_profile(seed_light_sheet_off_waist)
    profile_aslm = axial_profile(seed_aslm)
    z_um = -z_extent_um / 2 + (np.arange(psf_size_z) + 0.5) * dz

    ax_profile = fig.add_subplot(gs[2, :])
    # Each curve normalised to its own peak -- the plot compares *shape*, not
    # the energy redistribution the gate causes, so say so in the axis label
    # rather than presenting these as raw normalised intensities. Explicit
    # `color=CURVE_COLORS[...]` keeps this curve's colour identical to its
    # appearance in the sectioning and outside-core panels below.
    ax_profile.plot(
        z_um,
        profile_light_sheet / profile_light_sheet.max(),
        marker="o",
        markersize=3,
        color=CURVE_COLORS["light_sheet"],
        label="light sheet",
    )
    ax_profile.plot(
        z_um,
        profile_light_sheet_off_waist / profile_light_sheet_off_waist.max(),
        marker="o",
        markersize=3,
        color=CURVE_COLORS["light_sheet_off_waist"],
        label=f"light sheet ({off_waist_um:.1f} um off-waist)",
    )
    ax_profile.plot(
        z_um,
        profile_aslm / profile_aslm.max(),
        marker="o",
        markersize=3,
        color=CURVE_COLORS["aslm"],
        label=f"ASLM (slit width = {slit_width:.2f} um)",
    )
    ax_profile.axhline(0.5, color="gray", linestyle="--", linewidth=1.0)
    ax_profile.set_xlabel("Z (um)")
    ax_profile.set_ylabel("intensity, each curve peak-normalised to 1.0")
    ax_profile.set_title("Axial (Z) intensity profile through each seed's true peak")
    # Zoom to where the curves actually carry visible structure -- the full
    # volume extent (matching the projection panels above) squeezes the
    # half-max crossing separation the reader needs to see into a few pixels
    # of screen space. Bound derived from where any curve clears 2% of its
    # own peak, plus a fixed margin, rather than a hardcoded window.
    visible = (
        (profile_light_sheet / profile_light_sheet.max() >= 0.02)
        | (profile_light_sheet_off_waist / profile_light_sheet_off_waist.max() >= 0.02)
        | (profile_aslm / profile_aslm.max() >= 0.02)
    )
    visible_z = z_um[visible]
    margin = 2.0 * dz
    ax_profile.set_xlim(visible_z.min() - margin, visible_z.max() + margin)
    ax_profile.legend()

    # Plane-energy optical-sectioning panel (row 3). log y axis (energy spans
    # orders of magnitude away from focus); dashed vertical lines mark the
    # +-DOF window that `in_focus_energy_fraction` sums over; each legend
    # label carries its own measured in-focus fraction, computed from the
    # data, never hardcoded.
    energy_light_sheet = plane_energy_profile(seed_light_sheet)
    energy_light_sheet_off_waist = plane_energy_profile(seed_light_sheet_off_waist)
    energy_aslm = plane_energy_profile(seed_aslm)

    fraction_light_sheet = in_focus_energy_fraction(seed_light_sheet, dof_um, dz)
    fraction_light_sheet_off_waist = in_focus_energy_fraction(
        seed_light_sheet_off_waist, dof_um, dz
    )
    fraction_aslm = in_focus_energy_fraction(seed_aslm, dof_um, dz)

    ax_sectioning = fig.add_subplot(gs[3, :])
    ax_sectioning.semilogy(
        z_um,
        energy_light_sheet,
        marker="o",
        markersize=3,
        color=CURVE_COLORS["light_sheet"],
        label=f"light sheet (in-focus energy {fraction_light_sheet:.3f})",
    )
    ax_sectioning.semilogy(
        z_um,
        energy_light_sheet_off_waist,
        marker="o",
        markersize=3,
        color=CURVE_COLORS["light_sheet_off_waist"],
        label=(
            f"light sheet ({off_waist_um:.1f} um off-waist) "
            f"(in-focus energy {fraction_light_sheet_off_waist:.3f})"
        ),
    )
    ax_sectioning.semilogy(
        z_um,
        energy_aslm,
        marker="o",
        markersize=3,
        color=CURVE_COLORS["aslm"],
        label=(
            f"ASLM (slit width = {slit_width:.2f} um) "
            f"(in-focus energy {fraction_aslm:.3f})"
        ),
    )
    ax_sectioning.axvline(-dof_um, color="gray", linestyle="--", linewidth=1.0)
    ax_sectioning.axvline(dof_um, color="gray", linestyle="--", linewidth=1.0)
    ax_sectioning.set_xlabel("Z (um)")
    ax_sectioning.set_ylabel("plane energy fraction (log)")
    ax_sectioning.set_title(
        f"Plane energy vs Z (optical sectioning); dashed lines at +-DOF ({dof_um:.4f} um)"
    )
    ax_sectioning.set_xlim(visible_z.min() - margin, visible_z.max() + margin)
    ax_sectioning.legend(fontsize=8)

    # Outside-core-fraction panel (row 4): makes the "energy removed
    # off-waist" claim decidable by eye. Same z_um centring as the axial
    # profile above. r_core_um is `detection_core_radius_um`'s half-FWHM
    # radius, stated explicitly in the title.
    fraction_out_light_sheet = outside_core_fraction(seed_light_sheet, r_core_um, dxy)
    fraction_out_light_sheet_off_waist = outside_core_fraction(
        seed_light_sheet_off_waist, r_core_um, dxy
    )
    fraction_out_aslm = outside_core_fraction(seed_aslm, r_core_um, dxy)

    ax_fraction = fig.add_subplot(gs[4, :])
    ax_fraction.plot(
        z_um,
        fraction_out_light_sheet,
        marker="o",
        markersize=3,
        color=CURVE_COLORS["light_sheet"],
        label="light sheet",
    )
    ax_fraction.plot(
        z_um,
        fraction_out_light_sheet_off_waist,
        marker="o",
        markersize=3,
        color=CURVE_COLORS["light_sheet_off_waist"],
        label=f"light sheet ({off_waist_um:.1f} um off-waist)",
    )
    ax_fraction.plot(
        z_um,
        fraction_out_aslm,
        marker="o",
        markersize=3,
        color=CURVE_COLORS["aslm"],
        label=f"ASLM (slit width = {slit_width:.2f} um)",
    )
    ax_fraction.set_xlabel("Z (um)")
    ax_fraction.set_ylabel("fraction of plane energy outside core")
    ax_fraction.set_title(
        f"Lateral energy fraction outside the detection core "
        f"(core radius = {r_core_um:.4f} um, half the detection lateral FWHM) vs Z"
    )
    ax_fraction.set_xlim(visible_z.min() - margin, visible_z.max() + margin)
    ax_fraction.legend(fontsize=8)

    # The suptitle is a clean title with no citation or disclosure blurb. The
    # illumination-model and sampling/slit-mapping disclosures are in the
    # module docstring and printed to stdout by main() instead.
    fig.suptitle(
        f"Light sheet vs ASLM PSF seed comparison (ASLM slit width = {slit_width:.2f} um)",
        fontsize=11,
    )
    return fig


def main() -> None:
    """Generate light_sheet (waist/off-waist) and aslm PSF seeds, print summaries, and save a comparison figure."""
    _require_simulation_sampling(COMMON)
    describe_environment()

    seed_light_sheet = generate_psf_seed(
        psf_mode="light_sheet",
        illumination_beam=ILLUMINATION_BEAM,
        illumination_fill_factor=ILLUMINATION_FILL_FACTOR,
        **COMMON,
    )
    seed_light_sheet_off_waist = generate_psf_seed(
        psf_mode="light_sheet",
        emitter_offset=OFF_WAIST_UM,
        illumination_beam=ILLUMINATION_BEAM,
        illumination_fill_factor=ILLUMINATION_FILL_FACTOR,
        **COMMON,
    )
    seed_aslm = generate_psf_seed(
        psf_mode="aslm",
        slit_width=SLIT_WIDTH,
        illumination_beam=ILLUMINATION_BEAM,
        illumination_fill_factor=ILLUMINATION_FILL_FACTOR,
        **COMMON,
    )
    # seed_single supplies the fixed core radius (via detection_core_radius_um,
    # half the detection lateral FWHM) and the DOF (via axial_fwhm) that
    # in_focus_energy_fraction and the sectioning panel both use. The core
    # and DOF are defined from the unslit detection-only PSF, not from
    # either compared mode.
    # "single" has no illumination arm, so generate_psf_seed rejects both beam
    # keywords for it; it stays on COMMON alone.
    seed_single = generate_psf_seed(psf_mode="single", **COMMON)
    dxy = COMMON["dxy"]
    dz = COMMON["dz"]
    r_core_um = detection_core_radius_um(seed_single, dxy)
    dof_um = axial_fwhm(seed_single, dz)
    # Figure annotation: measured, never hardcoded, from the two
    # already-generated seeds -- see
    # measured_illumination_cross_section_fwhm_um's docstring.
    illumination_fwhm_um = measured_illumination_cross_section_fwhm_um(
        seed_light_sheet_off_waist, seed_single, dz
    )

    # Measured, never hardcoded, in-focus energy fractions for the printed
    # summary.
    fraction_light_sheet = in_focus_energy_fraction(seed_light_sheet, dof_um, dz)
    fraction_light_sheet_off_waist = in_focus_energy_fraction(
        seed_light_sheet_off_waist, dof_um, dz
    )
    fraction_aslm = in_focus_energy_fraction(seed_aslm, dof_um, dz)

    # The slit integrates along the pre-rotation propagation axis for every
    # direction -- there is no per-axis gate resolution and no full-extent
    # equivalence shortcut.
    propagation_window_um = COMMON["psf_size_z"] * COMMON["dz"]
    print(
        f"illumination_beam={ILLUMINATION_BEAM}, "
        f"illumination_fill_factor={ILLUMINATION_FILL_FACTOR}"
    )
    summarise_mode("light_sheet", seed_light_sheet, "none")
    summarise_mode(
        "light_sheet_off_waist",
        seed_light_sheet_off_waist,
        f"none, emitter {OFF_WAIST_UM} um off-waist",
    )
    summarise_mode(
        "aslm",
        seed_aslm,
        f"slit_width={SLIT_WIDTH}um sweep-integrated along propagation, "
        f"window={propagation_window_um:.3f}um",
    )
    print(
        "measured sheet thickness (illumination FWHM along Z) at the emitter, "
        f"{OFF_WAIST_UM:.1f} um off-waist: {illumination_fwhm_um:.3f} um"
    )
    print(f"detection DOF (axial FWHM of seed_single): {dof_um:.4f} um")
    print(f"detection core radius (half the detection lateral FWHM): {r_core_um:.4f} um")
    print(
        "in-focus energy fraction: "
        f"light_sheet waist={fraction_light_sheet:.3f}, "
        f"light_sheet {OFF_WAIST_UM:.1f}um off-waist={fraction_light_sheet_off_waist:.3f}, "
        f"aslm={fraction_aslm:.3f}"
    )
    # The illumination-model and sampling/slit-mapping disclosures are printed
    # here (and kept in the module docstring's Notes / limitations section),
    # not rendered into the figure.
    print(ILLUMINATION_MODEL_NOTE)
    print(SAMPLING_AND_SLIT_NOTE)

    fig = build_comparison_figure(
        seed_light_sheet,
        seed_light_sheet_off_waist,
        seed_aslm,
        COMMON,
        SLIT_WIDTH,
        off_waist_um=OFF_WAIST_UM,
        r_core_um=r_core_um,
        illumination_fwhm_um=illumination_fwhm_um,
        dof_um=dof_um,
    )

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "light_sheet_vs_aslm.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"wrote {output_path}")


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
