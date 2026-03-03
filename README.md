# near-extremal-threshold

Repository accompanying the manuscript:

**“Operational Extremality Bound from Near-Extremal Kerr Quasinormal Spectra”**  
Aleksey Buyanov  
ORCID: https://orcid.org/0009-0001-2621-9305  
Email: pppuuu7@gmail.com  

---

## Overview

This repository contains:

- Python analysis code used to generate all numerical figures and tables
  appearing in the manuscript.
- Scripts reproducing the Kerr QNM infrared scaling analysis.
- Scripts reproducing the Davies (GKLS) spectral gap scans.
- Generated LaTeX tables used directly in the paper.

The goal is full referee reproducibility from a clean checkout.

---

# 1. Quick Reproduction Guide (For Referees)

## Step 1 — Create virtual environment

Recommended: Python 3.11+

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
````

If you prefer not to install as editable, most scripts prepend `src/`
to `sys.path` automatically.

---

## Step 2 — Run full pipeline

```bash
python analysis/run_all.py
```

This regenerates:

* Enriched QNM bundle
* IR scaling exponent tables
* Ratio collapse figures
* LOFO stability tables
* Davies gap scan histograms
* LaTeX tables included in the manuscript

All outputs are written to:

```
figures/
```

---

# 2. Repository Structure

```
near-extremal-threshold/
│
├── src/nearextremal_threshold/
│     Core numerical models (Davies generator, baths, utilities)
│
├── analysis/
│     Reproducibility scripts
│     run_all.py  ← main entry point
│
├── data/
│     Input QNM dataset(s)
│
├── figures/
│     Generated figures and LaTeX tables used in the paper
│
└── README.md
```

---

# 3. Key Paper Results and Corresponding Scripts

## 3.1 IR Scaling Exponent ρ

Script:

```bash
python analysis/qnm_rho_by_family.py
```

Generates:

```
figures/qnm_absolute_tables/rho_by_family.png
figures/qnm_absolute_tables/qnm_abs_scaling_summary.tex
```

---

## 3.2 Ratio Collapse

Script:

```bash
python analysis/qnm_ratio_collapse.py
```

Generates:

```
figures/qnm_ratio_collapse/ratio_collapse_median_band.png
figures/qnm_ratio_collapse/ratio_collapse_binned_table.tex
```

---

## 3.3 Davies Gap Scan (Numerical Support of Theorem 1)

Script examples:

```bash
python analysis/davies_gap_scan.py --bath ohmic
python analysis/davies_gap_scan.py --bath super_p3
```

Full reproduction via:

```bash
python analysis/run_all.py
```

Generates:

```
figures/qnm_markov_gap/gap_scan_slope_hist_ohmic_like.png
figures/qnm_markov_gap/gap_scan_slope_hist_super_p3.png
```

---

# 4. Reproducibility Notes

* Random seeds are fixed where applicable.
* Linear algebra may vary slightly across BLAS/LAPACK builds.
* Paper uses robust statistics (medians, percentile bands, LOFO tests).
* Small floating-point variations do not affect conclusions.

---

# 5. License Information

This repository contains:

1. **Software (analysis code)**
2. **Scientific manuscript**

These are licensed separately.

---

## 5.1 Software License

Dual License Model:

### Academic & Non-Profit Use → FREE

Permitted for:

* Universities
* Public research institutions
* Individual non-commercial researchers

Conditions:

* Proper attribution required
* License text must be included
* Modifications must be indicated
* Redistribution only under same license terms

---

### Commercial Use → Requires Separate License

Any for-profit, SaaS, corporate R&D,
or revenue-generating use requires a separate written agreement.

Commercial inquiries:
[pppuuu7@gmail.com](mailto:pppuuu7@gmail.com)

Full license text:
See `LICENSE` file in this repository.

---

## 5.2 Manuscript License

The scientific manuscript is licensed under:

Creative Commons Attribution–NonCommercial 4.0
(CC BY-NC 4.0)

You may:

* Share
* Adapt

For non-commercial purposes with attribution.

Commercial use requires explicit permission.

See:
MANUSCRIPT_LICENSE.txt

---

# 6. Citation

If this repository is used in academic work, please cite the associated manuscript.

BibTeX entry will be added upon journal publication.

---

# 7. Author Information

Aleksey Buyanov
Independent Researcher
ORCID: [https://orcid.org/0009-0001-2621-9305](https://orcid.org/0009-0001-2621-9305)
Email: [pppuuu7@gmail.com](mailto:pppuuu7@gmail.com)

---

# 8. Contact

For reproducibility questions or clarification:
[pppuuu7@gmail.com](mailto:pppuuu7@gmail.com)

For commercial licensing:
[pppuuu7@gmail.com](mailto:pppuuu7@gmail.com)
