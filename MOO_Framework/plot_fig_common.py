"""Shared print-layout helpers for the plot scripts.

Every figure is designed at a fixed single-column print width (16 cm - the
A4 text width with 2.5 cm margins) at 300 dpi, so each complete figure can be
dropped into a paper at that width without rescaling or deformation (insert
at fixed width, keep the aspect ratio).

``save_both`` also exports a ``*_notext`` twin of every figure: pure graphics
with all text removed (titles, axis labels, tick numbers, legends, colourbar
ticks/titles and data annotations), for venues that add their own labelling.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import matplotlib.figure
import matplotlib
import matplotlib.text

# Consistent manuscript typography across uplift, violin, and 3D figures.
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

TARGET_WIDTH_CM = 16.0  # A4 single-column text width (21.0 cm - 2 * 2.5 cm margins)
DPI = 300


def read_current_result(path: Path) -> dict:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parent
    region = root / "data/regions/Stage_SC_Regions.json"
    current_hash = hashlib.sha256(region.read_bytes()).hexdigest()
    if (data.get("region_sha256") != current_hash
            or data.get("source") != "Sionna RadioMapSolver surrogate"
            or len(data.get("receivers", [])) != 600
            or data.get("coverage", {}).get("receivers") != 600):
        raise ValueError(f"Stale or incomplete 600-point result: {path}")
    stage = next((part for part in path.parts if part in {"frame", "enclosure"}), None)
    if stage is None:
        raise ValueError(f"Cannot identify construction stage for: {path}")
    pm_state = json.loads((root / "result" / stage / "workspace/bim/pipeline_state.json").read_text(encoding="utf-8"))
    pm_problem = json.loads(Path(pm_state["paths"]["wap_optimization"]["problem"]).read_text(encoding="utf-8"))
    profile = pm_problem["optimization_problem"]["sensor_profile"]
    if profile["key"] != "SP3" or profile["tx_power_dbm"] != 23.0:
        raise ValueError(f"{stage} results use an old SP profile; rerun MOO with SP3 before plotting")
    if path.parents[1].name in {"EbM1", "EbM2", "EbM3", "GcM1", "GcM2"}:
        manifest_path = path.parents[1] / "case_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("surrogate_site_projection") is not False
                or manifest.get("region_sha256") != current_hash
                or manifest.get("sensor_profile") != "SP3"
                or manifest.get("tx_power_dbm") != 23.0):
            raise ValueError(f"Baseline has not been scored at its original sites: {path}")
    return data


def cm_to_in(cm: float) -> float:
    return cm / 2.54


def strip_text(fig: matplotlib.figure.Figure) -> None:
    """Remove every visible text artist from the figure (pure-graphics version).

    Titles are removed through the public API where available; the figure
    suptitle is handled through the guarded private attribute because older
    matplotlib versions expose no public remover, and a 3D/colourbar axis may
    not implement every labelling setter.
    """
    for ax in fig.get_axes():
        try:
            ax.set_title("")
        except Exception:
            pass
        try:
            ax.set_xlabel("")
            ax.set_ylabel("")
        except Exception:
            pass
        for label_setter in ("set_zlabel", "set_zticklabels"):
            setter = getattr(ax, label_setter, None)
            if setter is not None:
                try:
                    setter("" if label_setter == "set_zlabel" else [])
                except Exception:
                    pass
        try:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        except Exception:
            pass
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        for text in list(getattr(ax, "texts", [])):
            text.remove()
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_text("")
        suptitle.set_visible(False)
    for text in list(getattr(fig, "texts", [])):
        text.remove()


def _leftover_texts(fig: matplotlib.figure.Figure) -> list[str]:
    """Any visible non-empty text left after strip_text - for the save_both check."""
    found: list[str] = []
    for ax in fig.get_axes():
        if ax.get_title():
            found.append(f"title {ax.get_title()!r}")
        if ax.get_xlabel():
            found.append(f"xlabel {ax.get_xlabel()!r}")
        if ax.get_ylabel():
            found.append(f"ylabel {ax.get_ylabel()!r}")
        if hasattr(ax, "get_zlabel") and ax.get_zlabel():
            found.append(f"zlabel {ax.get_zlabel()!r}")
        for tl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            if tl.get_visible() and tl.get_text():
                found.append(f"ticklabel {tl.get_text()!r}")
        for text in getattr(ax, "texts", []):
            if text.get_visible() and text.get_text():
                found.append(f"axtext {text.get_text()!r}")
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                if text.get_text():
                    found.append(f"legend {text.get_text()!r}")
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_visible() and suptitle.get_text():
        found.append(f"suptitle {suptitle.get_text()!r}")
    for text in getattr(fig, "texts", []):
        if text.get_visible() and text.get_text():
            found.append(f"figtext {text.get_text()!r}")
    return found


def save_both(fig: matplotlib.figure.Figure, path: Path, dpi: int = DPI) -> Path:
    """Save the labelled figure, then its *_notext pure-graphics twin.

    Returns the *_notext path; prints a check that no text survived stripping.
    """
    path = Path(path)
    wrong_fonts = [
        (artist.get_text(), artist.get_fontproperties().get_name())
        for artist in fig.findobj(match=matplotlib.text.Text)
        if artist.get_visible() and artist.get_text().strip()
        and artist.get_fontproperties().get_name() != "Times New Roman"
    ]
    if wrong_fonts:
        raise ValueError(f"Figure text is not Times New Roman: {wrong_fonts[:5]}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    strip_text(fig)
    notext_path = path.with_name(f"{path.stem}_notext{path.suffix}")
    fig.savefig(notext_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    leftover = _leftover_texts(fig)
    print(f"notext check {notext_path.name}: "
          + ("clean" if not leftover else "LEFTOVER TEXT: " + ", ".join(leftover)))
    return notext_path
