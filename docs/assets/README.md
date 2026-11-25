# Research assets

The software license covers the page implementation. Paper PDFs, recorded observations and dataset-derived imagery retain their applicable research-asset terms.

| Path | Content and source |
|---|---|
| `papers/FedVista.pdf` | Anonymous main manuscript, unchanged. |
| `papers/FedVista_supplement.pdf` | Original anonymous supplementary PDF, unchanged. |
| `media/02_piper_real_and_gs.*` | Original PiPER Real/GS recording. The original head and external camera pairs are stacked without the former outer white canvas. 900 × 720, full 15.52-second duration and original 60 fps. |
| `figures/overview.svg` | Native reconstruction of `last-update.pdf`, the method figure used in the paper. Original vector geometry, glyph outlines and observation patches are retained. |
| `figures/*.svg` | Native reconstructions of the paper's experiment and visual-world figures. Coordinates, labels, curves and measurements are unchanged. |
| `figures/sources.json` | Source figure names, hashes and vector-element inventory. |
| `images/overview.svg`, `images/overview.png` | Static overview previews for the README and link sharing. |
| `images/favicon.svg` | FedVista icon. |

The browser renders the figures as inline SVG elements. Dark and Light apply to the whole page. CSS variables change vector colors; the Light theme restores the original figure palette. Photographic observations retain their original pixels. Animated overlays follow the method figure's existing paths and do not alter the underlying figure.

Regenerate the SVGs with `scripts/build_paper_figures.py --source /path/to/paper/figures` (PyMuPDF required). The source path is an input only and is not embedded in released files. The result panels preserve their original protocol distinctions: three-task reference outcomes, five-round progress, development-stage variants, and local optimization traces.

The PiPER task-stage viewer uses the four paired groups in `Q04_piper_real_3dgs_sequence.pdf`. On small screens, the Replica viewer shows two adjacent scenes and the method viewer focuses on the selected stage; neither changes the original drawing coordinates. Numerical chart animations reveal original points, curves, bars, distributions and cells, then remove all transient styles. Reduced-motion preferences show the complete plots directly.
