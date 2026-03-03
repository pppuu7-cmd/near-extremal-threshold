#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures_list_tex.py

Generate a RevTeX-friendly figures_list.tex and a manifest.json from an explicit list of figure files.
Uses optional caption helper txt files when present.

Default figure list (edit as needed):
  - figures/qnm_ratio_collapse/ratio_collapse_median_band.png
  - figures/qnm_absolute_tables/rho_by_family.png

Optional inputs it will use if they exist:
  - figures/qnm_ratio_collapse/ratio_collapse_caption.txt
  - figures/qnm_absolute_tables/abstract_one_liner.txt
  - figures/qnm_ratio_collapse/ratio_collapse_inference.tex  (this is a table, not a figure; we include it separately)
  - figures/qnm_ratio_collapse/ratio_collapse_binned_table.tex (table)

Outputs:
  - figures/figures_list.tex
  - figures/figures_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict


DEFAULT_FIGS = [
    "figures/qnm_ratio_collapse/ratio_collapse_median_band.png",
    "figures/qnm_absolute_tables/rho_by_family.png",
]

DEFAULT_TABLE_INPUTS = [
    "figures/qnm_ratio_collapse/ratio_collapse_binned_table.tex",
    "figures/qnm_ratio_collapse/ratio_collapse_inference.tex",
]


def safe_makedirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_text_if_exists(p: Path) -> str | None:
    if p.exists():
        t = p.read_text(encoding="utf-8").strip()
        return t if t else None
    return None


def latex_escape(s: str) -> str:
    # minimal escape for common chars in captions
    repl = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def guess_label_from_path(p: Path) -> str:
    stem = p.stem
    # Make a stable label: fig:<stem>
    # Replace any odd chars with underscore
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem)
    return f"fig:{safe}"


def guess_caption(p: Path) -> str:
    # Map known figures to more informative captions.
    # You can edit these anytime.
    s = str(p).replace("\\", "/")
    if s.endswith("ratio_collapse_median_band.png"):
        # try to use caption txt if exists
        return (
            r"Near-extremal collapse of $R(\Delta)=|\mathrm{Im}(\omega M)|/(T_H M)$ versus "
            r"$\Delta=1-\chi^2$ (pooled $\Delta\le 0.19$). Median and 16--84\% band shown."
        )
    if s.endswith("rho_by_family.png"):
        return (
            r"Universality of the scaling exponent $\rho$ in "
            r"$|\mathrm{Im}(\omega M)|\propto (T_H M)^{\rho}$ across Kerr QNM families."
        )
    return f"Figure: {p.name}"


def build_figure_block(fig_path: Path, caption: str, label: str, width: str = r"\linewidth") -> str:
    cap = latex_escape(caption)
    lab = latex_escape(label)
    path_str = fig_path.as_posix()
    return "\n".join(
        [
            r"\begin{figure}[t]",
            r"\centering",
            rf"\includegraphics[width={width}]{{{path_str}}}",
            rf"\caption{{{cap}}}",
            rf"\label{{{lab}}}",
            r"\end{figure}",
            "",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate figures_list.tex (explicit list) + manifest.json.")
    ap.add_argument("--figs", type=str, default=",".join(DEFAULT_FIGS),
                    help="Comma-separated list of figure image paths.")
    ap.add_argument("--tables", type=str, default=",".join(DEFAULT_TABLE_INPUTS),
                    help="Comma-separated list of .tex table inputs to mention in manifest.")
    ap.add_argument("--out_tex", type=str, default=str(Path("figures") / "figures_list.tex"))
    ap.add_argument("--out_manifest", type=str, default=str(Path("figures") / "figures_manifest.json"))
    args = ap.parse_args()

    fig_list = [Path(s.strip()) for s in args.figs.split(",") if s.strip()]
    table_list = [Path(s.strip()) for s in args.tables.split(",") if s.strip()]

    out_tex = Path(args.out_tex)
    out_manifest = Path(args.out_manifest)
    safe_makedirs(out_tex.parent)

    # Optional captions from txt
    ratio_caption_txt = Path("figures/qnm_ratio_collapse/ratio_collapse_caption.txt")
    ratio_caption = read_text_if_exists(ratio_caption_txt)

    one_liner_txt = Path("figures/qnm_absolute_tables/abstract_one_liner.txt")
    abstract_one_liner = read_text_if_exists(one_liner_txt)

    blocks: List[str] = []
    manifest: Dict[str, object] = {
        "figures": [],
        "tables": [],
        "notes": {},
    }

    if abstract_one_liner:
        manifest["notes"]["abstract_one_liner"] = abstract_one_liner

    for fp in fig_list:
        if not fp.exists():
            raise SystemExit(f"[error] missing figure file: {fp}")

        label = guess_label_from_path(fp)
        cap = guess_caption(fp)

        # If this is the collapse plot, prefer the generated caption txt if present
        if fp.name == "ratio_collapse_median_band.png" and ratio_caption:
            # ratio_caption is already a sentence; just use it
            cap = ratio_caption

        blocks.append(build_figure_block(fp, cap, label, width=r"\linewidth"))
        manifest["figures"].append(
            {
                "path": fp.as_posix(),
                "label": label,
                "caption": cap,
            }
        )

    # Tables: we don't embed them as figures; we record them in manifest for \input.
    for tp in table_list:
        if not tp.exists():
            # not fatal: maybe user didn't generate it yet
            continue
        manifest["tables"].append(
            {
                "path": tp.as_posix(),
                "input": rf"\input{{{tp.as_posix()}}}",
            }
        )

    tex_content = "\n".join(blocks)
    out_tex.write_text(tex_content, encoding="utf-8")
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[ok] wrote {out_tex}")
    print(f"[ok] wrote {out_manifest}")
    if abstract_one_liner:
        print("\n[info] abstract one-liner captured into manifest (notes.abstract_one_liner).")


if __name__ == "__main__":
    main()