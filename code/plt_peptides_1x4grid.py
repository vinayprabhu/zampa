"""Publication-ready 1 x 4 AMP structure figure for BayLearn/NeurIPS.

Colab setup
------------
!pip -q install "numpy>=1.24" "scipy>=1.10" "matplotlib>=3.7"

Example
-------
from plt_peptides_1x4grid import plt_peptides_1x4grid

plt_peptides_1x4grid(
    [gpt_sequence, edison_sequence, abacus_sequence, claude_sequence],
    [gpt_mic, edison_mic, abacus_mic, claude_mic],
    "baylearn_amp_structures",
    model_names=["GPT", "Edison", "Abacus", "Claude"],
)

This creates ``baylearn_amp_structures.png`` and
``baylearn_amp_structures.pdf``, displays the figure in a notebook, and
returns a dictionary containing the paths and confidence summaries. The
default visual places a quiet, translucent atom-and-bond model beneath a glossy
pLDDT ribbon, adds subtle projected shadows and coordinate triads, and includes
a shared confidence colorbar. Pass
``background="transparent"`` for transparent exports.

Important scientific caveat
---------------------------
ESMFold predicts a single protein-like conformer from sequence. Short AMPs are
often flexible and membrane/solvent dependent, so these panels are qualitative
structural hypotheses, not evidence of a unique solution structure. The figure
therefore colors the trace by pLDDT and reports mean pLDDT explicitly.
"""

from __future__ import annotations

import hashlib
import math
import numbers
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.interpolate import CubicSpline


ESMFOLD_API = "https://api.esmatlas.com/foldSequence/v1/pdb/"
DEFAULT_MODEL_NAMES = ("GPT", "Edison", "Abacus", "Claude")
# Retained for notebooks that imported the old constant directly.
DEFAULT_LABELS = DEFAULT_MODEL_NAMES
VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

# AlphaFold/ESMFold pLDDT colors: very low, low, confident, very high.
PLDDT_BOUNDS = np.array([0.0, 50.0, 70.0, 90.0, 100.0])
PLDDT_CMAP = ListedColormap(["#FF7D45", "#FFDB13", "#65CBF3", "#0053D6"])
PLDDT_NORM = BoundaryNorm(PLDDT_BOUNDS, PLDDT_CMAP.N, clip=True)

# NGL-inspired chainbow: warm N terminus -> cool C terminus. The custom stops
# avoid muddy mid-tones when the figure is reduced to NeurIPS column width.
CHAINBOW_CMAP = LinearSegmentedColormap.from_list(
    "amp_chainbow",
    [
        "#D90B3D",
        "#F03B35",
        "#FF8A1F",
        "#F6D746",
        "#68C96B",
        "#19B9B1",
        "#1676D2",
        "#5123A5",
    ],
)


@dataclass(frozen=True)
class FoldedTrace:
    """C-alpha trace and confidence parsed from an ESMFold PDB response."""

    coordinates: np.ndarray  # [L, 3]
    plddt: np.ndarray  # [L]
    pdb_text: str


def _validate_inputs(
    peptide_sequences: Sequence[str],
    mic_scores: Sequence[object],
    model_names: Sequence[str],
) -> tuple[list[str], list[object], list[str]]:
    if len(peptide_sequences) != 4 or len(mic_scores) != 4:
        raise ValueError("peptide_sequences and mic_scores must each contain exactly 4 items.")
    if len(model_names) != 4:
        raise ValueError("model_names must contain exactly 4 items.")

    cleaned: list[str] = []
    for i, sequence in enumerate(peptide_sequences):
        if not isinstance(sequence, str):
            raise TypeError(f"Sequence {i + 1} must be a string.")
        sequence = re.sub(r"\s+", "", sequence).upper()
        invalid = sorted(set(sequence) - VALID_AA)
        if invalid:
            raise ValueError(
                f"Sequence {i + 1} contains non-canonical residue(s): {', '.join(invalid)}"
            )
        if not 8 <= len(sequence) <= 50:
            raise ValueError(
                f"Sequence {i + 1} has length {len(sequence)}; this benchmark expects 8-50 aa."
            )
        cleaned.append(sequence)

    cleaned_names = [str(name).strip() for name in model_names]
    if any(not name for name in cleaned_names):
        raise ValueError("model_names cannot contain blank names.")

    return cleaned, list(mic_scores), cleaned_names


def _output_stem(output_filename: str | Path) -> Path:
    path = Path(output_filename).expanduser()
    if path.suffix.lower() in {".png", ".pdf"}:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _fold_with_esm_atlas(
    sequence: str,
    cache_dir: Path,
    timeout_seconds: int,
) -> FoldedTrace:
    """Fold one sequence with the public ESM Atlas endpoint and cache its PDB."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()[:20]
    pdb_path = cache_dir / f"esmfold_{digest}.pdb"

    if pdb_path.exists():
        pdb_text = pdb_path.read_text(encoding="utf-8")
    else:
        request = Request(
            ESMFOLD_API,
            data=sequence.encode("ascii"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                pdb_text = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                "ESMFold request failed. Check network access and retry; already cached "
                "predictions remain usable."
            ) from exc
        if not any(line.startswith("ATOM") for line in pdb_text.splitlines()):
            raise RuntimeError("ESMFold returned no ATOM records; no structure can be plotted.")
        pdb_path.write_text(pdb_text, encoding="utf-8")

    coordinates, plddt = _parse_ca_trace(pdb_text)
    return FoldedTrace(coordinates=coordinates, plddt=plddt, pdb_text=pdb_text)


def _parse_ca_trace(pdb_text: str) -> tuple[np.ndarray, np.ndarray]:
    coordinates: list[list[float]] = []
    confidence: list[float] = []
    seen_residues: set[tuple[str, str, str]] = set()

    for line in pdb_text.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        altloc = line[16:17]
        if altloc not in {" ", "A"}:
            continue
        residue_key = (line[21:22], line[22:26], line[26:27])
        if residue_key in seen_residues:
            continue
        seen_residues.add(residue_key)
        try:
            coordinates.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
            confidence.append(float(line[60:66]))
        except ValueError as exc:
            raise RuntimeError("Malformed coordinates in ESMFold PDB response.") from exc

    xyz = np.asarray(coordinates, dtype=float)
    plddt = np.asarray(confidence, dtype=float)
    if xyz.ndim != 2 or xyz.shape[0] < 2 or xyz.shape[1] != 3:
        raise RuntimeError("Could not recover a usable C-alpha trace from the PDB response.")
    if np.nanmax(plddt) <= 1.5:
        plddt = plddt * 100.0
    return xyz, np.clip(plddt, 0.0, 100.0)


def _pca_transform(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic center and 3D rotation derived from a C-alpha trace."""
    center = coordinates.mean(axis=0, keepdims=True)
    centered = coordinates - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rotation = vh.T
    oriented = centered @ rotation

    # Make N -> C run left-to-right; the camera still sees all three dimensions.
    if oriented[-1, 0] < oriented[0, 0]:
        rotation[:, 0] *= -1
        oriented[:, 0] *= -1
    if np.sum(oriented[:, 1] * np.arange(len(oriented))) < 0:
        rotation[:, 1] *= -1
    return center, rotation


def _pca_orient(coordinates: np.ndarray) -> np.ndarray:
    center, rotation = _pca_transform(coordinates)
    return (coordinates - center) @ rotation


def _parse_heavy_atoms(
    pdb_text: str,
) -> tuple[np.ndarray, list[str], list[str], list[tuple[str, int, str]]]:
    """Parse non-hydrogen atoms needed for a compact ball-and-stick overlay."""
    coordinates: list[list[float]] = []
    elements: list[str] = []
    atom_names: list[str] = []
    residue_keys: list[tuple[str, int, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        altloc = line[16:17]
        if altloc not in {" ", "A"}:
            continue
        atom_name = line[12:16].strip()
        element = line[76:78].strip().upper()
        if not element:
            element = next((char for char in atom_name if char.isalpha()), "C").upper()
        if element == "H":
            continue
        key = (line[21:22], line[22:26], line[26:27], atom_name)
        if key in seen:
            continue
        seen.add(key)
        try:
            coordinates.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
            residue_keys.append((line[21:22], int(line[22:26]), line[26:27]))
        except ValueError:
            continue
        elements.append(element)
        atom_names.append(atom_name)

    return np.asarray(coordinates, dtype=float), elements, atom_names, residue_keys


def _infer_bonds(
    coordinates: np.ndarray,
    elements: Sequence[str],
    residue_keys: Sequence[tuple[str, int, str]],
) -> np.ndarray:
    """Infer heavy-atom covalent bonds from distances within neighboring residues."""
    radii = {"C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07}
    bonds: list[tuple[int, int]] = []
    for i in range(len(coordinates)):
        chain_i, residue_i, _ = residue_keys[i]
        for j in range(i + 1, len(coordinates)):
            chain_j, residue_j, _ = residue_keys[j]
            if chain_i != chain_j or abs(residue_i - residue_j) > 1:
                continue
            distance = float(np.linalg.norm(coordinates[i] - coordinates[j]))
            cutoff = 1.24 * (radii.get(elements[i], 0.76) + radii.get(elements[j], 0.76))
            if 0.75 < distance <= cutoff:
                bonds.append((i, j))
    return np.asarray(bonds, dtype=int)


def _smooth_trace(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    samples_per_residue: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate in arc-length coordinates without changing residue anchors."""
    step = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    step = np.maximum(step, 1e-6)
    arc = np.r_[0.0, np.cumsum(step)]
    dense_arc = np.linspace(arc[0], arc[-1], max(40, len(arc) * samples_per_residue))

    if len(coordinates) >= 4:
        dense_xyz = CubicSpline(arc, coordinates, axis=0, bc_type="natural")(dense_arc)
        dense_conf = CubicSpline(arc, confidence, bc_type="natural")(dense_arc)
    else:
        dense_xyz = np.column_stack(
            [np.interp(dense_arc, arc, coordinates[:, dim]) for dim in range(3)]
        )
        dense_conf = np.interp(dense_arc, arc, confidence)
    return dense_xyz, np.clip(dense_conf, 0.0, 100.0)


def _parallel_transport_frames(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct stable normal/binormal vectors along a 3D centerline."""
    tangents = np.gradient(coordinates, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)

    normals = np.empty_like(tangents)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(seed, tangents[0]))) > 0.85:
        seed = np.array([0.0, 1.0, 0.0])
    normal = seed - np.dot(seed, tangents[0]) * tangents[0]
    normals[0] = normal / max(np.linalg.norm(normal), 1e-12)

    for i in range(1, len(tangents)):
        normal = normals[i - 1] - np.dot(normals[i - 1], tangents[i]) * tangents[i]
        length = np.linalg.norm(normal)
        if length < 1e-8:
            fallback = np.cross(tangents[i - 1], tangents[i])
            if np.linalg.norm(fallback) < 1e-8:
                fallback = np.array([0.0, 1.0, 0.0])
            normal = fallback - np.dot(fallback, tangents[i]) * tangents[i]
            length = np.linalg.norm(normal)
        normals[i] = normal / max(length, 1e-12)

    binormals = np.cross(tangents, normals)
    binormals /= np.maximum(np.linalg.norm(binormals, axis=1, keepdims=True), 1e-12)
    return normals, binormals


def _tube_collection(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    radius: float,
    sides: int = 10,
) -> Poly3DCollection:
    """Create a shaded, pLDDT-colored molecular tube around a C-alpha trace."""
    normals, binormals = _parallel_transport_frames(coordinates)
    theta = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    rings = (
        coordinates[:, None, :]
        + radius
        * (
            np.cos(theta)[None, :, None] * normals[:, None, :]
            + np.sin(theta)[None, :, None] * binormals[:, None, :]
        )
    )

    faces: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    light = np.array([-0.45, -0.35, 0.82])
    light /= np.linalg.norm(light)

    for i in range(len(rings) - 1):
        base = np.asarray(PLDDT_CMAP(PLDDT_NORM((confidence[i] + confidence[i + 1]) / 2.0)))
        for j in range(sides):
            j2 = (j + 1) % sides
            face = np.array([rings[i, j], rings[i + 1, j], rings[i + 1, j2], rings[i, j2]])
            face_normal = np.cross(face[1] - face[0], face[3] - face[0])
            face_normal /= max(np.linalg.norm(face_normal), 1e-12)
            illumination = 0.58 + 0.42 * max(float(np.dot(face_normal, light)), 0.0)
            rgba = base.copy()
            rgba[:3] = np.clip(rgba[:3] * illumination + 0.06, 0.0, 1.0)
            faces.append(face)
            colors.append(rgba)

    collection = Poly3DCollection(
        faces,
        facecolors=colors,
        edgecolors="none",
        linewidths=0.0,
        antialiased=True,
        zsort="average",
    )
    # Keep molecular geometry as paths in PDF/SVG output. Rasterization was the
    # reason earlier PDFs looked soft when zoomed or placed in LaTeX.
    collection.set_rasterized(False)
    return collection


def _ribbon_collection(
    coordinates: np.ndarray,
    confidence: np.ndarray,
    half_width: float,
    half_thickness: float,
    color_by: str,
    opacity: float,
) -> Poly3DCollection:
    """Build a glossy, solid cartoon ribbon around a smooth C-alpha trace."""
    normals, binormals = _parallel_transport_frames(coordinates)
    left_top = coordinates + half_width * normals + half_thickness * binormals
    right_top = coordinates - half_width * normals + half_thickness * binormals
    left_bottom = coordinates + half_width * normals - half_thickness * binormals
    right_bottom = coordinates - half_width * normals - half_thickness * binormals

    if color_by == "rainbow":
        point_values = np.linspace(0.0, 1.0, len(coordinates))
        point_colors = CHAINBOW_CMAP(point_values)
    elif color_by == "plddt":
        point_colors = PLDDT_CMAP(PLDDT_NORM(confidence))
    else:
        raise ValueError("color_by must be 'rainbow' or 'plddt'.")

    faces: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    light = np.array([-0.42, -0.55, 0.72], dtype=float)
    light /= np.linalg.norm(light)

    def add_face(vertices: np.ndarray, base: np.ndarray, ambient: float = 0.48) -> None:
        normal = np.cross(vertices[1] - vertices[0], vertices[3] - vertices[0])
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        diffuse = abs(float(np.dot(normal, light)))
        illumination = ambient + (1.0 - ambient) * diffuse
        rgba = np.asarray(base, dtype=float).copy()
        rgba[:3] = np.clip(rgba[:3] * illumination + 0.10 * diffuse, 0.0, 1.0)
        rgba[3] = opacity
        faces.append(vertices)
        colors.append(rgba)

    for i in range(len(coordinates) - 1):
        base = (point_colors[i] + point_colors[i + 1]) / 2.0
        # Broad faces read as a ribbon; narrow faces supply depth and gloss.
        add_face(np.array([left_top[i], left_top[i + 1], right_top[i + 1], right_top[i]]), base, 0.60)
        add_face(
            np.array([right_bottom[i], right_bottom[i + 1], left_bottom[i + 1], left_bottom[i]]),
            base,
            0.34,
        )
        add_face(np.array([left_bottom[i], left_bottom[i + 1], left_top[i + 1], left_top[i]]), base, 0.42)
        add_face(
            np.array([right_top[i], right_top[i + 1], right_bottom[i + 1], right_bottom[i]]),
            base,
            0.42,
        )

    # Close both termini so the ribbon looks like a solid molecular object.
    add_face(
        np.array([left_bottom[0], left_top[0], right_top[0], right_bottom[0]]),
        point_colors[0],
        0.55,
    )
    add_face(
        np.array([right_bottom[-1], right_top[-1], left_top[-1], left_bottom[-1]]),
        point_colors[-1],
        0.55,
    )

    collection = Poly3DCollection(
        faces,
        facecolors=colors,
        edgecolors="none",
        linewidths=0.0,
        antialiased=True,
        zsort="average",
    )
    # Matplotlib can emit every ribbon face as vector geometry. This makes the
    # PDF larger than a rasterized hybrid, but it remains sharp at any scale.
    collection.set_rasterized(False)
    return collection


def _add_soft_shadow(
    ax,
    coordinates: np.ndarray,
    ground_z: float,
    opacity: float,
) -> None:
    """Add a restrained vector shadow projected onto a horizontal ground plane."""
    if opacity <= 0.0 or len(coordinates) < 2:
        return
    projected = np.asarray(coordinates, dtype=float).copy()
    projected[:, 2] = ground_z
    segments = np.stack([projected[:-1], projected[1:]], axis=1)
    # Several faint vector strokes imitate a soft penumbra without embedding a
    # bitmap in the PDF.
    for width, strength in ((5.6, 0.16), (3.2, 0.24), (1.45, 0.38)):
        shadow = Line3DCollection(
            segments,
            colors=[(0.04, 0.10, 0.18, opacity * strength)],
            linewidths=width,
            capstyle="round",
            zorder=0,
        )
        shadow.set_rasterized(False)
        ax.add_collection3d(shadow)


def _add_coordinate_triad(
    ax,
    center: np.ndarray,
    display_span: np.ndarray,
    opacity: float = 0.66,
) -> None:
    """Draw a small x/y/z triad in the PCA-oriented molecular coordinate frame."""
    lower = center - display_span / 2.0
    origin = lower + np.array([0.11, 0.12, 0.12]) * display_span
    length = 0.13 * float(np.max(display_span))
    colors = ("#D64B4B", "#35A36F", "#3578C7")
    labels = ("x", "y", "z")
    vectors = np.eye(3) * length
    for vector, color, label in zip(vectors, colors, labels):
        ax.quiver(
            *origin,
            *vector,
            color=color,
            alpha=opacity,
            linewidth=0.55,
            arrow_length_ratio=0.24,
            normalize=False,
            zorder=7,
        )
        endpoint = origin + 1.18 * vector
        ax.text(
            *endpoint,
            label,
            color=color,
            alpha=min(1.0, opacity + 0.12),
            fontsize=3.4,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=8,
        )


def _format_mic(value: object, mic_unit: str) -> str:
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        numeric = float(value)
        rendered = "NA" if not math.isfinite(numeric) else f"{numeric:.3g}"
    else:
        rendered = str(value).strip()
    return f"MIC  {rendered} {mic_unit}".rstrip()


def _confidence_label(plddt: np.ndarray) -> str:
    mean = float(np.nanmean(plddt))
    if mean >= 90:
        word = "very high"
    elif mean >= 70:
        word = "confident"
    elif mean >= 50:
        word = "low"
    else:
        word = "very low"
    return f"mean pLDDT {mean:.1f}  |  {word}"


def _draw_panel(
    ax,
    folded: FoldedTrace,
    sequence: str,
    mic_score: object,
    label: str,
    panel_letter: str,
    mic_unit: str,
    color_by: str,
    show_atoms: bool,
    ribbon_opacity: float,
    atom_opacity: float,
    bond_opacity: float,
    atom_scale: float,
    show_axes: bool,
    show_shadow: bool,
    shadow_opacity: float,
    background: str,
) -> None:
    pca_center, pca_rotation = _pca_transform(folded.coordinates)
    xyz = (folded.coordinates - pca_center) @ pca_rotation
    dense_xyz, dense_plddt = _smooth_trace(xyz, folded.plddt)
    ca_spacing = float(np.median(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
    # A slightly narrower ribbon leaves side-chain chemistry legible when the
    # atom-and-bond layer is enabled.
    width_fraction = 0.205 if show_atoms else 0.25
    ribbon_half_width = float(np.clip(width_fraction * ca_spacing, 0.66, 1.05))
    ribbon_half_thickness = float(np.clip(0.055 * ca_spacing, 0.16, 0.25))

    ax.set_facecolor("none" if background == "transparent" else "white")
    ax.patch.set_alpha(0.0 if background == "transparent" else 1.0)
    ax.set_proj_type("persp", focal_length=1.08)
    ax.view_init(elev=24, azim=-58)
    ax.set_axis_off()

    # Parse the complete heavy-atom geometry before rendering so the axis limits
    # include side chains instead of clipping them at the card edges.
    atom_xyz, elements, _, residue_keys = _parse_heavy_atoms(folded.pdb_text)
    if len(atom_xyz):
        atom_xyz = (atom_xyz - pca_center) @ pca_rotation

    extent_xyz = np.vstack([dense_xyz, atom_xyz]) if show_atoms and len(atom_xyz) else dense_xyz
    center = (extent_xyz.min(axis=0) + extent_xyz.max(axis=0)) / 2.0
    raw_span = np.maximum(np.ptp(extent_xyz, axis=0), 1e-6)
    max_span = max(float(raw_span.max()), 4.0)
    display_span = np.maximum(raw_span + 2.9 * ribbon_half_width, 0.38 * max_span)
    for setter, c, span in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center, display_span):
        setter(c - span / 2.0, c + span / 2.0)
    ax.set_box_aspect(display_span, zoom=1.26)

    if show_shadow:
        ground_z = center[2] - 0.42 * display_span[2]
        _add_soft_shadow(ax, dense_xyz, ground_z, shadow_opacity)

    # Draw chemistry first and the ribbon last. Together with low opacity and
    # reduced marker sizes, this makes atoms/bonds informative context rather
    # than the visually dominant object.
    if show_atoms and len(atom_xyz):
        atom_palette = {
            "C": "#667483",
            "N": "#2F6FDB",
            "O": "#E05252",
            "S": "#D7A914",
            "P": "#D9822B",
        }
        bonds = _infer_bonds(atom_xyz, elements, residue_keys)
        if len(bonds):
            bond_segments: list[np.ndarray] = []
            bond_colors: list[str] = []
            for i, j in bonds:
                midpoint = (atom_xyz[i] + atom_xyz[j]) / 2.0
                bond_segments.extend(
                    [np.array([atom_xyz[i], midpoint]), np.array([midpoint, atom_xyz[j]])]
                )
                bond_colors.extend(
                    [atom_palette.get(elements[i], "#8291A5"), atom_palette.get(elements[j], "#8291A5")]
                )
            sticks = Line3DCollection(
                bond_segments,
                colors=bond_colors,
                linewidths=0.46,
                alpha=bond_opacity,
                capstyle="round",
                zorder=2,
            )
            sticks.set_rasterized(False)
            ax.add_collection3d(sticks)

        element_array = np.asarray(elements)
        for element in dict.fromkeys(elements):
            mask = element_array == element
            size = atom_scale * (
                8.2 if element in {"S", "P"} else 6.3 if element in {"N", "O"} else 3.2
            )
            element_alpha = atom_opacity * (0.68 if element == "C" else 1.0)
            ax.scatter(
                atom_xyz[mask, 0], atom_xyz[mask, 1], atom_xyz[mask, 2],
                s=size,
                c=atom_palette.get(element, "#8291A5"),
                edgecolors="white",
                linewidths=0.12,
                depthshade=True,
                alpha=element_alpha,
                rasterized=False,
                zorder=3,
            )

    ribbon = _ribbon_collection(
        dense_xyz,
        dense_plddt,
        ribbon_half_width,
        ribbon_half_thickness,
        color_by,
        ribbon_opacity,
    )
    ribbon.set_zorder(5)
    ax.add_collection3d(ribbon)

    if show_axes:
        _add_coordinate_triad(ax, center, display_span)

    # Card annotations remain vector-sharp in the exported PDF.
    ax.text2D(
        0.025, 0.970, panel_letter, transform=ax.transAxes,
        ha="left", va="top", fontsize=6.8, fontweight="bold", color="#0E2340",
    )
    ax.text2D(
        0.105, 0.970, label, transform=ax.transAxes,
        ha="left", va="top", fontsize=6.8, fontweight="bold", color="#0E2340",
    )
    ax.text2D(
        0.025, 0.058, sequence, transform=ax.transAxes,
        ha="left", va="bottom", fontsize=4.15, family="DejaVu Sans Mono",
        color="#324A66",
    )
    ax.text2D(
        0.975, 0.970, _format_mic(mic_score, mic_unit), transform=ax.transAxes,
        ha="right", va="top", fontsize=5.45, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.29,rounding_size=0.14", fc="#0E2F53", ec="none"),
    )
    ax.text2D(
        0.975, 0.016, f"L={len(sequence)}  |  pLDDT {np.nanmean(folded.plddt):.0f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=4.0,
        color="#5D6F83",
    )


def _make_figure(
    folded_structures: Sequence[FoldedTrace],
    sequences: Sequence[str],
    mic_scores: Sequence[object],
    model_names: Sequence[str],
    mic_unit: str,
    color_by: str,
    show_atoms: bool,
    ribbon_opacity: float,
    atom_opacity: float,
    bond_opacity: float,
    atom_scale: float,
    show_axes: bool,
    show_shadow: bool,
    shadow_opacity: float,
    background: str,
    show_plddt_colorbar: bool,
) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 9,
            "axes.unicode_minus": False,
        }
    )
    # 5.48 in fits inside the NeurIPS 2023 template's 5.5 in text block.
    figure_color = "none" if background == "transparent" else "white"
    # A slim footer holds a single shared pLDDT legend without shrinking any
    # molecular panel or breaking the requested 1 x 4 structure layout.
    has_colorbar = show_plddt_colorbar and color_by == "plddt"
    fig = plt.figure(
        figsize=(5.48, 1.60 if has_colorbar else 1.46),
        facecolor=figure_color,
    )
    fig.patch.set_alpha(0.0 if background == "transparent" else 1.0)
    grid = fig.add_gridspec(
        1,
        4,
        left=0.003,
        right=0.997,
        bottom=0.130 if has_colorbar else 0.008,
        top=0.992,
        wspace=0.014,
    )

    for i, (folded, sequence, mic, label) in enumerate(
        zip(folded_structures, sequences, mic_scores, model_names)
    ):
        # Explicit z-order keeps the translucent chemical model beneath the
        # cartoon ribbon instead of letting mplot3d promote it unpredictably.
        ax = fig.add_subplot(grid[0, i], projection="3d", computed_zorder=False)
        _draw_panel(
            ax,
            folded,
            sequence,
            mic,
            label,
            chr(65 + i),
            mic_unit,
            color_by,
            show_atoms,
            ribbon_opacity,
            atom_opacity,
            bond_opacity,
            atom_scale,
            show_axes,
            show_shadow,
            shadow_opacity,
            background,
        )

    if has_colorbar:
        # ESMFold follows AlphaFold's four confidence bins. A discrete shared
        # bar is both more truthful and more legible than four tiny legends.
        cax = fig.add_axes([0.410, 0.065, 0.320, 0.028])
        cax.set_facecolor("none" if background == "transparent" else "white")
        scalar_map = mpl.cm.ScalarMappable(norm=PLDDT_NORM, cmap=PLDDT_CMAP)
        scalar_map.set_array([])
        colorbar = fig.colorbar(
            scalar_map,
            cax=cax,
            orientation="horizontal",
            boundaries=PLDDT_BOUNDS,
            ticks=PLDDT_BOUNDS,
            spacing="proportional",
        )
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(
            axis="x", which="both", length=1.5, width=0.35, pad=0.7,
            labelsize=4.0, colors="#43566C",
        )
        fig.text(
            0.398, 0.079, "pLDDT confidence", ha="right", va="center",
            fontsize=4.8, fontweight="bold", color="#253B53",
        )
        for boundary in PLDDT_BOUNDS[1:-1]:
            colorbar.ax.axvline(boundary, color="white", linewidth=0.45)
    return fig


def plt_peptides_1x4grid(
    peptide_sequences: Sequence[str],
    mic_scores: Sequence[object],
    output_filename: str | Path,
    *,
    model_names: Sequence[str] = DEFAULT_MODEL_NAMES,
    system_labels: Sequence[str] | None = None,
    mic_unit: str = "µM",
    cache_dir: str | Path | None = None,
    timeout_seconds: int = 300,
    dpi: int = 600,
    display: bool = True,
    background: str = "white",
    color_by: str = "plddt",
    show_plddt_colorbar: bool = True,
    show_atoms: bool = True,
    ribbon_opacity: float = 0.90,
    atom_opacity: float = 0.46,
    bond_opacity: float = 0.28,
    atom_scale: float = 0.78,
    show_axes: bool = True,
    show_shadow: bool = True,
    shadow_opacity: float = 0.20,
) -> dict[str, object]:
    """Predict, display, and export a compact one-row, four-column AMP figure.

    Parameters
    ----------
    peptide_sequences
        Exactly four canonical amino-acid sequences, each 8-50 residues long,
        in the same order as ``model_names``.
    mic_scores
        Exactly four MIC values/labels in the same order as the sequences.
    output_filename
        Filename stem or a name ending in .png/.pdf. Both formats are written.
    model_names
        Exactly four model names in the same order as the sequences and MICs.
        The legacy keyword ``system_labels`` remains accepted as an alias.

    Other parameters are keyword-only. ``background`` accepts ``"white"`` or
    ``"transparent"``. ``color_by`` accepts the default ``"plddt"`` or
    ``"rainbow"``. The shared pLDDT colorbar appears when ``color_by="plddt"``
    and ``show_plddt_colorbar=True``. Heavy atoms and inferred bonds are shown
    beneath the ribbon by default. ``ribbon_opacity``, ``atom_opacity``,
    ``bond_opacity``, and ``atom_scale`` tune the hierarchy. ``show_axes`` and
    ``show_shadow`` add restrained depth cues. PNG is high-resolution for
    inspection; PDF keeps molecular geometry, shadows, and text as vectors.
    """
    if system_labels is not None:
        if tuple(model_names) != DEFAULT_MODEL_NAMES:
            raise ValueError("Pass only model_names; system_labels is a legacy alias.")
        model_names = system_labels

    sequences, mic_values, labels = _validate_inputs(
        peptide_sequences, mic_scores, model_names
    )
    if background not in {"white", "transparent"}:
        raise ValueError("background must be 'white' or 'transparent'.")
    if color_by not in {"rainbow", "plddt"}:
        raise ValueError("color_by must be 'rainbow' or 'plddt'.")
    for name, value in {
        "ribbon_opacity": ribbon_opacity,
        "atom_opacity": atom_opacity,
        "bond_opacity": bond_opacity,
        "shadow_opacity": shadow_opacity,
    }.items():
        if not isinstance(value, numbers.Real) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be a number between 0 and 1.")
    if not isinstance(atom_scale, numbers.Real) or isinstance(atom_scale, bool) or atom_scale <= 0:
        raise ValueError("atom_scale must be a positive number.")
    if not isinstance(show_axes, bool) or not isinstance(show_shadow, bool):
        raise TypeError("show_axes and show_shadow must be booleans.")
    stem = _output_stem(output_filename)
    structure_cache = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else stem.parent / ".esmfold_cache"
    )

    unique_folds: dict[str, FoldedTrace] = {}
    for sequence in dict.fromkeys(sequences):
        unique_folds[sequence] = _fold_with_esm_atlas(
            sequence, structure_cache, timeout_seconds
        )
    folded = [unique_folds[sequence] for sequence in sequences]

    fig = _make_figure(
        folded,
        sequences,
        mic_values,
        labels,
        mic_unit,
        color_by,
        show_atoms,
        float(ribbon_opacity),
        float(atom_opacity),
        float(bond_opacity),
        float(atom_scale),
        show_axes,
        show_shadow,
        float(shadow_opacity),
        background,
        show_plddt_colorbar,
    )
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    transparent = background == "transparent"
    save_color = "none" if transparent else "white"
    fig.savefig(png_path, dpi=dpi, facecolor=save_color, transparent=transparent)
    fig.savefig(pdf_path, facecolor=save_color, transparent=transparent)

    if display:
        try:
            from IPython.display import display as notebook_display

            notebook_display(fig)
        except ImportError:
            plt.show()
    plt.close(fig)

    mean_plddt = [float(np.nanmean(item.plddt)) for item in folded]
    if any(value < 50 for value in mean_plddt):
        warnings.warn(
            "At least one mean pLDDT is below 50. Treat that panel as a likely "
            "flexible/disordered structural hypothesis, not a stable fold.",
            stacklevel=2,
        )

    return {
        "png": str(png_path.resolve()),
        "pdf": str(pdf_path.resolve()),
        "mean_plddt": dict(zip(labels, mean_plddt)),
    }


def generate_amp_structure_figure(*args, **kwargs) -> dict[str, object]:
    """Backward-compatible alias for earlier notebooks."""
    return plt_peptides_1x4grid(*args, **kwargs)


__all__ = ["plt_peptides_1x4grid", "generate_amp_structure_figure"]
